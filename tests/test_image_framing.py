import hashlib
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

    def assert_face_visible(self, width, height, face):
        result = framing.position_for_faces(width, height, [face])
        self.assertEqual(result['reason'], 'primary_face')
        crop_w, crop_h = min(width, height * 16 / 9), min(height, width * 9 / 16)
        left = (width - crop_w) * result['x'] / 100
        top = (height - crop_h) * result['y'] / 100
        x, y, w, h, _ = face
        # CSS percentages are rounded; allow less than a hundredth of a pixel.
        self.assertLessEqual(left, x + .01)
        self.assertLessEqual(top, y + .01)
        self.assertGreaterEqual(left + crop_w, x + w - .01)
        self.assertGreaterEqual(top + crop_h, y + h - .01)
        self.assertTrue(all(0 <= result[k] <= 100 for k in ('x', 'y')))
        return result, (left, top, crop_w, crop_h)

    def test_portrait_reduces_padding_without_losing_face_or_headroom(self):
        face = [110, 140, 180, 200, .96]
        result, (_, top, _, crop_h) = self.assert_face_visible(400, 800, face)
        self.assertLess(result['y'], 50)
        headroom = face[1] - top
        footroom = top + crop_h - face[1] - face[3]
        self.assertGreater(headroom, footroom)
        self.assertGreater(footroom, 0)

    def test_landscape_reduces_horizontal_padding(self):
        result, _ = self.assert_face_visible(1200, 400, [550, 50, 600, 180, .96])
        self.assertGreater(result['x'], 50)

    def test_face_exactly_filling_crop_does_not_require_padding(self):
        result, _ = self.assert_face_visible(400, 800, [100, 20, 200, 225, .96])
        self.assertLess(result['y'], 50)

    def test_padding_near_each_edge_keeps_the_full_face(self):
        for width, height, face in (
                (400, 800, [100, 0, 200, 200, .96]),
                (400, 800, [100, 600, 200, 200, .96]),
                (1200, 400, [0, 100, 650, 150, .96]),
                (1200, 400, [550, 100, 650, 150, .96])):
            with self.subTest(width=width, height=height, face=face):
                self.assert_face_visible(width, height, face)

    def test_horizontal_face_too_large_still_falls_back(self):
        result = framing.position_for_faces(1200, 400, [[100, 50, 800, 180, .96]])
        self.assertEqual(result['reason'], 'face_exceeds_frame')

    def test_padding_does_not_override_ambiguity_or_low_confidence(self):
        result = framing.position_for_faces(400, 800,
            [[0, 40, 180, 200, .96], [200, 420, 180, 200, .96]])
        self.assertEqual(result['reason'], 'ambiguous_faces')
        self.assertEqual(framing.position_for_faces(400, 800,
            [[110, 140, 180, 200, .89]])['reason'], 'no_clear_face')

    def test_old_padding_fallback_cache_cannot_hide_the_new_crop(self):
        url = 'https://images.unsplash.com/test'
        old = {'x': 50, 'y': 50, 'reason': 'face_exceeds_frame', 'version': 'yunet-2023mar-v1'}
        expected = framing.position_for_faces(400, 800, [[110, 140, 180, 200, .96]])
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / (hashlib.sha256((old['version'] + url).encode()).hexdigest() + '.json')
            old_path.write_text(json.dumps(old), encoding='utf-8')
            with patch.object(framing, 'download_image', return_value=b'photo') as download:
                with patch.object(framing, 'analyze_image', return_value=expected) as detect:
                    self.assertEqual(framing.frame_image(url, cache_dir=directory), expected)
                    self.assertEqual(framing.frame_image(url, cache_dir=directory), expected)
                    download.assert_called_once()
                    detect.assert_called_once()
            self.assertEqual(json.loads(old_path.read_text(encoding='utf-8')), old)

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

    def test_wikimedia_thumbnail_host_reaches_download(self):
        with patch.object(framing.requests, 'get', side_effect=requests.Timeout) as get:
            with self.assertRaises(requests.Timeout):
                framing.download_image('https://thumb.wikimedia.org/wikipedia/commons/thumb/a/photo.jpg')
            get.assert_called_once()

    def test_disallowed_host_never_requested(self):
        with patch.object(framing.requests, 'get') as get:
            with self.assertRaises(ValueError):
                framing.download_image('https://example.com/photo.jpg')
            get.assert_not_called()

    def test_template_old_and_new_issues(self):
        from main import templates
        env = templates.env
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
