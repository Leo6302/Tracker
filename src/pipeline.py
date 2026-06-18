import cv2
import numpy as np
import tempfile
from pathlib import Path

from .tracking.tracker import VideoTracker
from .prediction.trainer import ModelManager
from .prediction.preprocessor import build_sequences, denormalize
from .visualization.renderer import FrameRenderer, get_color
from .visualization.exporter import VideoExporter, CSVExporter, SummaryImageExporter, build_track_summary


def _build_analyzers(analysis_config, fps, fw, fh):
    """Instantiate requested analyzer objects from analysis_config dict."""
    analyzers = []
    if not analysis_config:
        return analyzers

    lines_cfg = analysis_config.get('counting_lines', [])
    if lines_cfg and analysis_config.get('enable_traffic'):
        from .analysis.counting_line import CountingLineSet
        analyzers.append(CountingLineSet(lines_cfg))

    scale_mpp = analysis_config.get('scale_mpp')
    if (scale_mpp and analysis_config.get('enable_speed')
            and analysis_config.get('enable_traffic')):
        from .analysis.speed_estimator import SpeedEstimator
        analyzers.append(SpeedEstimator(scale_mpp))

    zones = analysis_config.get('zones', [])
    if zones and analysis_config.get('enable_od') and analysis_config.get('enable_traffic'):
        from .analysis.od_matrix import ODMatrixBuilder
        analyzers.append(ODMatrixBuilder(zones))

    if analysis_config.get('enable_heatmap') and analysis_config.get('enable_urban'):
        from .analysis.density_heatmap import DensityHeatmapBuilder
        analyzers.append(DensityHeatmapBuilder(
            fw, fh, analysis_config.get('heatmap_classes')
        ))

    if zones and analysis_config.get('enable_urban'):
        from .analysis.zone_analyzer import ZoneAnalyzer
        analyzers.append(ZoneAnalyzer(zones, analysis_config.get('zone_areas', {})))

    if (scale_mpp and analysis_config.get('enable_speed')
            and analysis_config.get('enable_traffic')):
        from .analysis.congestion import CongestionDetector
        analyzers.append(CongestionDetector(scale_mpp))

    return analyzers


class TrackingPipeline:
    def __init__(self, config: dict, lstm_device, yolo_device: str = "cpu"):
        self.config = config
        self.tracker = VideoTracker(config, yolo_device=yolo_device)
        self.model_manager = ModelManager(config, lstm_device)
        self.renderer = FrameRenderer()

    def load_model(self, pretrained_path) -> bool:
        return self.model_manager.load_pretrained(pretrained_path)

    # ------------------------------------------------------------------
    # Fine-tune helpers
    # ------------------------------------------------------------------

    def _collect_finetune_data(self, video_path, fraction, class_filter):
        seq_len = self.config['seq_len']
        pred_len = self.config['pred_len']
        window = seq_len + pred_len
        fw = fh = 1

        temp = VideoTracker(self.config, yolo_device=self.tracker.yolo_device)
        for frame_idx, _, _, (fw, fh), total_frames in temp.process_video(
            video_path, class_filter=class_filter
        ):
            if frame_idx >= int(total_frames * fraction):
                break

        sequences, targets = [], []
        for hist in temp.track_history.values():
            hist_list = list(hist)
            if len(hist_list) < window:
                continue
            arr = np.array(
                [[p[1] / fw, p[2] / fh, p[3] / fw, p[4] / fh] for p in hist_list],
                dtype=np.float32,
            )
            for start in range(len(arr) - window + 1):
                sequences.append(arr[start: start + seq_len])
                targets.append(arr[start + seq_len: start + window])
        return sequences, targets

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, video_path, class_filter=None, lstm_mode='pretrained',
            progress_callback=None, analysis_config=None, uncertainty=False):
        """Process video end-to-end.

        Returns a dict with all output paths and analysis results.
        """
        seq_len = self.config['seq_len']
        pred_len = self.config['pred_len']

        def _progress(pct, msg=""):
            if progress_callback:
                progress_callback(pct, 100, msg)

        if lstm_mode == 'finetune':
            _progress(0, "파인튜닝 데이터 수집 중...")
            seqs, tgts = self._collect_finetune_data(
                video_path,
                self.config.get('finetune_fraction', 0.3),
                class_filter,
            )
            _progress(10, "LSTM 파인튜닝 중...")
            self.model_manager.finetune(
                seqs, tgts,
                epochs=self.config.get('finetune_epochs', 5),
                lr=self.config.get('lr', 0.001),
            )

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        analyzers = _build_analyzers(analysis_config or {}, fps, fw, fh)
        # Find specific analyzers for renderer overlays (by name)
        counting_set = next((a for a in analyzers if a.name == 'counting_lines'), None)
        zone_az = next((a for a in analyzers if a.name == 'zone_analysis'), None)
        heatmap_az = next((a for a in analyzers if a.name == 'heatmap'), None)

        tmp_dir = Path(tempfile.mkdtemp())
        video_out = tmp_dir / "annotated.mp4"
        csv_out = tmp_dir / "predictions.csv"
        img_out = tmp_dir / "trajectory_summary.png"

        vid_writer = VideoExporter(video_out, fps, (fw, fh))
        csv_exp = CSVExporter(pred_len)
        img_exp = SummaryImageExporter(fw, fh)

        bg_frame = None
        stats = {'total_tracks': set(), 'total_frames': 0}

        for frame_idx, frame, track_data, (fw, fh), total_frames in \
                self.tracker.process_video(video_path, class_filter=class_filter):
            stats['total_frames'] = total_frames
            if bg_frame is None:
                bg_frame = frame.copy()

            predictions: dict = {}
            for td in track_data:
                tid = td['track_id']
                stats['total_tracks'].add(tid)

                seq_tensor = build_sequences(td['history'], seq_len, fw, fh)
                pred_px = None
                pred_std_px = None
                if seq_tensor is not None:
                    if uncertainty:
                        pred_norm, std_norm = \
                            self.model_manager.predict_with_uncertainty(seq_tensor)
                        pred_px = denormalize(pred_norm, fw, fh)
                        pred_std_px = std_norm * np.array([fw, fh, fw, fh],
                                                          dtype=np.float32)
                    else:
                        pred_norm = self.model_manager.predict(seq_tensor)
                        pred_px = denormalize(pred_norm, fw, fh)
                    predictions[tid] = pred_px

                bgr = get_color(tid)
                rgb = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
                img_exp.update(tid, td['cx'], td['cy'], rgb,
                               pred=pred_px[:, :2].tolist() if pred_px is not None else None)
                csv_exp.add(frame_idx, tid, td['cls_name'],
                            td['cx'], td['cy'], td['w'], td['h'],
                            pred_px, pred_std=pred_std_px)

            for az in analyzers:
                az.update(frame_idx, track_data, fps)

            annotated = self.renderer.draw(frame, track_data, predictions)
            if counting_set:
                self.renderer.draw_counting_lines(annotated, counting_set)
            if zone_az:
                self.renderer.draw_zones(annotated, zone_az)
            vid_writer.write_frame(annotated)

            pct = int((frame_idx + 1) / max(total_frames, 1) * 100)
            _progress(pct, f"처리 중... {frame_idx+1}/{total_frames} 프레임")

        vid_writer.release()
        csv_exp.save(csv_out)
        img_exp.save(img_out, bg_frame=bg_frame)

        stats['total_tracks'] = len(stats['total_tracks'])
        duration_s = stats['total_frames'] / max(fps, 1e-6)

        # Finalize all analyzers
        analysis_results = {}
        for az in analyzers:
            analysis_results[az.name] = az.finalize()

        track_summary = build_track_summary(
            csv_exp.rows,
            scale_mpp=(analysis_config or {}).get('scale_mpp'),
            speed_data=analysis_results.get('speed'),
        )

        # Save density heatmap
        heatmap_img_out = None
        if heatmap_az:
            heatmap_img_out = tmp_dir / "density_heatmap.png"
            heatmap_az.save_heatmap(heatmap_img_out, bg_frame=bg_frame)

        # Excel export
        excel_out = None
        exp_fmt = (analysis_config or {}).get('export_format', 'CSV만')
        if exp_fmt in ('CSV + Excel', 'CSV + Excel + 차트'):
            try:
                from .visualization.stats_exporter import ResearchExporter
                excel_out = tmp_dir / "research_report.xlsx"
                ok = ResearchExporter(
                    csv_exp.rows, pred_len, analysis_results,
                    self.config, duration_s, video_path,
                    track_summary=track_summary
                ).save(excel_out)
                if not ok:
                    excel_out = None
            except Exception as e:
                print(f"[Excel export error] {e}")
                excel_out = None

        # Charts export
        charts_out = None
        if exp_fmt == 'CSV + Excel + 차트':
            try:
                from .visualization.stats_exporter import PlotlyExporter
                charts_out = tmp_dir / "research_charts.html"
                ok = PlotlyExporter(analysis_results, duration_s).save_html(charts_out)
                if not ok:
                    charts_out = None
            except Exception as e:
                print(f"[Charts export error] {e}")
                charts_out = None

        return {
            'video': str(video_out),
            'csv': str(csv_out),
            'trajectory_img': str(img_out),
            'stats': stats,
            'analysis': analysis_results,
            'heatmap_img': str(heatmap_img_out) if heatmap_img_out else None,
            'excel': str(excel_out) if excel_out else None,
            'charts': str(charts_out) if charts_out else None,
            'duration_s': duration_s,
            'track_summary': track_summary,
        }
