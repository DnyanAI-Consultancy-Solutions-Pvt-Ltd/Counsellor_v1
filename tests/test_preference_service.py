import unittest

from services.preference_service import PreferenceService


class PreferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = PreferenceService()
        self.rows = [
            {"college": "ABC College, Pune", "branch": "Computer Engineering", "location": "Pune", "zone": "Target", "evidence_ids": [1]},
            {"college": "XYZ Institute, Mumbai", "branch": "IT", "location": "Mumbai", "zone": "Safe", "evidence_ids": [2]},
        ]

    def test_remove_all_location_matches(self):
        updated, removed = self.service.remove_matching(self.rows, field="location", value="Pune")
        self.assertEqual(len(removed), 1)
        self.assertEqual(self.service.count_matching(updated, field="location", value="Pune"), 0)

    def test_add_exact_unique_count(self):
        candidates = [
            {"college": "New One", "branch": "CSE", "zone": "Dream", "evidence_ids": [3]},
            {"college": "New Two", "branch": "AI", "zone": "Target", "evidence_ids": [4]},
        ]
        updated, added = self.service.add_exact(self.rows, candidates, 2)
        self.assertEqual(len(added), 2)
        self.assertEqual(len(updated), 4)


if __name__ == "__main__":
    unittest.main()
