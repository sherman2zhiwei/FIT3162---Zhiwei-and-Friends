import sys
sys.path.append('aspect-aggregation')

import unittest
from asp_agg_utils import _readXML, _add6PosFeautures, _flatten, _oneHotVectorize, _performance_measure, _clean_text

class TestFunc(unittest.TestCase):

    def setUp(self):
        """
        This function is to create some examples used for testing
        """
        pass


    def test_readXML(self):
        pass

    def test_add6PosFeatures(self):
        pass

    def test_flatten(self):
        """
        Test function _flatten
        """
        self.assertEqual(_flatten([[1], [2]]), [1, 2])
        self.assertEqual(_flatten([["a","b"], ["c"]]), ["a", "b", "c"])
        self.assertFalse(_flatten([[[1]], [[2]]]) == [[1], 2]) # Output should be [[1], [2]]
        self.assertTrue(_flatten([[1], ["a"], [2], ["b"]]), [1, "a", 2, "b"])
        self.assertTrue(_flatten([[]]) == [])

    def test_oneHotVectorize(self):
        pass

    def test_performance_measure(self):
        pass

    def test_clean_text(self):
        """
        Test function _clean_text
        """
        # test for lowercase
        self.assertEqual(_clean_text("ABC"), "abc")
        # test for stopword removal
        self.assertEqual(_clean_text("the food is nice."), "food nice .")
        # test for removing quotations that surround words
        self.assertEqual(_clean_text("'food' 'is' 'nice'"), "food nice")
        # test for contraction he's, she's and it's (if not expanded, then "'s" will not be treated as a stopword)
        self.assertEqual(_clean_text("He's a good boy and she's a good girl. It's not a good dog."), \
                                     "good boy good girl . good dog .")
        # test for contraction can't, 'll, n't (if not expanded, then they might not be treated as a stopword)
        self.assertEqual(_clean_text("I don't think I can't win the game but I'll lose him if you didn't ask me."), \
                                     "think win game lose ask .")
        # test for contraction i'm and 're (if not expanded, then "'s" will not be treated as a stopword)
        self.assertEqual(_clean_text("I'm not going to that place but you're going to that place."), \
                                     "going place going place .")
        # test for multiple consecutive whitespace removal
        self.assertEqual(_clean_text("dim     sum     is     good"), \
                                     "dim sum good")
        # test for starting and ending whitespace removal
        self.assertEqual(_clean_text(" dim sum is good "), \
                                     "dim sum good")


if __name__ == "__main__":
    unittest.main()
