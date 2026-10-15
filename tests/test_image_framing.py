import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import requests

import image_framing as framing


class FramingTests(unittest.TestCase):
    def test_portrait_keeps_headroom(self):
        result = framing.position_for_faces(400, 800, [[160, 60, 80, 100, .95]])
        top = (800 - 225) * result['y'] / 100
        self.assertLessEqual(top, 20)
        self.assertGreaterEqual(top + 225, 175)

    def test_landscape_moves_horizontally(self):
        result = framing.position_for_faces(1200, 400, [[1050, 100, 80, 100, .95]])
        self.assertEqual(result['x'], 100)
        self.assertEqual(result['y'], 50)

    def test_already_visible_face_stays_centered(self):
        result = framing.position_for_faces(400, 800, [[160, 350, 80, 100, .95]])
        self.assertEqual((result['x'], result['y']), (50, 50))

    def test_ambiguous_group_stays_centered(self):
        result = framing.position_for_faces(400, 800,
            [[30, 40, 80, 100, .99], [200, 400, 75, 95, .99]])
        self.assertEqual(result['reason'], 'ambiguous_faces')

    def test_low_confidence_and_invalid_boxes_ignored(self):
        for face in ([100, 40, 80, 100, .89], [-10, 40, 80, 100, .99],
                     [100, float('nan'), 80, 100, .99]):
            self.assertEqual(framing.position_for_faces(400, 800, [face])['reason'],
                             'no_clear_face')

    def test_face_too_large_for_frame_falls_back(self):
        self.assertEqual(framing.position_for_faces(400, 800,
            [[80, 20, 200, 300, .99]])['reason'], 'face_exceeds_frame')

    def test_real_detector_on_blank_image(self):
        _, encoded = cv2.imencode('.jpg', np.zeros((300, 400, 3), dtype=np.uint8))
        self.assertEqual(framing.analyze_image(encoded.tobytes())['reason'], 'no_clear_face')

    def test_cache_avoids_repeat_download_and_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(framing, 'download_image', return_value=b'photo') as download:
                with patch.object(framing, 'analyze_image', return_value=framing.centered('no_clear_face')) as detect:
                    first = framing.frame_image('https://images.unsplash.com/test', cache_dir=directory)
                    self.assertEqual(framing.frame_image('https://images.unsplash.com/test', cache_dir=directory), first)
                    download.assert_called_once()
                    detect.assert_called_once()

    def test_network_failure_does_not_cache_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(framing, 'download_image', side_effect=requests.Timeout):
                result = framing.frame_image('https://images.unsplash.com/test', cache_dir=directory)
            self.assertEqual(result['reason'], 'analysis_unavailable')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_image_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(framing.frame_image('test', content=b'bad', cache_dir=directory)['reason'],
                             'analysis_unavailable')

    def test_disallowed_host_never_requested(self):
        with patch.object(framing.requests, 'get') as get:
            with self.assertRaises(ValueError):
                framing.download_image('https://example.com/photo.jpg')
            get.assert_not_called()

    def test_template_old_and_new_issues(self):
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(str(framing.ROOT / 'templates')))
        topic = {'topic_title': 'Test', 'image_url': 'https://example.com/image.jpg'}
        template = env.get_template('index.html')
        # Use the actual template to catch compatibility with old saved issues.
        for position in (None, {'x': 50, 'y': 0}):
            if position is not None:
                topic['image_framing'] = position
            html = template.render(summaries=[topic], request=None, market_data=None)
            self.assertIn(topic['image_url'], html)
            if position is not None:
                self.assertIn('object-position: 50.0% 0.0%', html)


if __name__ == '__main__':
    unittest.main()
