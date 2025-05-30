"""
スクレイピングユーティリティのテスト
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.scraping_utils import RaceScraper, parse_weight, generate_race_ids


class TestRaceScraper(unittest.TestCase):
    """RaceScraperクラスのテスト"""
    
    def setUp(self):
        """テストのセットアップ"""
        self.scraper = RaceScraper()
        
    def test_init(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.scraper.user_agents)
        self.assertEqual(len(self.scraper.failed_urls), 0)
        
    @patch('requests.get')
    def test_fetch_with_retry_success(self, mock_get):
        """正常なリクエストのテスト"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_get.return_value = mock_response
        
        result = self.scraper.fetch_with_retry('http://test.com')
        self.assertEqual(result, b'test content')
        
    @patch('requests.get')
    def test_fetch_with_retry_404(self, mock_get):
        """404エラーのテスト"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.scraper.fetch_with_retry('http://test.com/404')
        self.assertIsNone(result)
        self.assertIn('http://test.com/404', self.scraper.failed_urls)
        
    def test_parse_weight_normal(self):
        """正常な体重パースのテスト"""
        weight, diff = parse_weight('480(+2)')
        self.assertEqual(weight, 480)
        self.assertEqual(diff, 2)
        
    def test_parse_weight_negative(self):
        """マイナス体重差のテスト"""
        weight, diff = parse_weight('476(-4)')
        self.assertEqual(weight, 476)
        self.assertEqual(diff, -4)
        
    def test_parse_weight_invalid(self):
        """無効な体重文字列のテスト"""
        weight, diff = parse_weight('invalid')
        self.assertEqual(weight, 0)
        self.assertEqual(diff, 0)
        
    def test_generate_race_ids(self):
        """レースID生成のテスト"""
        place_dict = {"01": "札幌", "05": "東京"}
        race_ids = generate_race_ids(2023, place_dict)
        
        # 2場所 × 6開催 × 9日 × 12レース = 1296
        self.assertEqual(len(race_ids), 1296)
        
        # 形式チェック
        self.assertTrue(all(len(rid) == 12 for rid in race_ids))
        self.assertTrue(all(rid.startswith('2023') for rid in race_ids))


if __name__ == '__main__':
    unittest.main()