import unittest
from godnyas_default_0b import load_default_0b


class AssetTest(unittest.TestCase):
    def test_strict_0b(self):
        result = load_default_0b().respond("ベンチ回して")
        self.assertEqual(result["status"], "DONE")
        self.assertEqual(result["trainable_parameters"], 0)
        self.assertEqual(result["model_calls"], 0)

    def test_unknown_abstains(self):
        self.assertEqual(load_default_0b().respond("未知の宇宙法則を作って")["status"], "UNRESOLVED")

    def test_reasoning(self):
        self.assertEqual(load_default_0b().reason("(12 + 5) * 3")["answer"], 51)


if __name__ == "__main__":
    unittest.main()
