import unittest

from benchmark_experiment import BenchmarkExperiment


class BenchmarkExperimentTest(unittest.TestCase):

  def test_to_dict(self):
    be = BenchmarkExperiment("some name", "cpu", "PJRT", "some xla_flags",
                             "openxla", "train", "123")
    actual = be.to_dict()
    self.assertEqual(8, len(actual))
    self.assertEqual("some name", actual["experiment_name"])
    self.assertEqual("cpu", actual["accelerator"])
    self.assertTrue("accelerator_model" in actual)
    self.assertEqual("PJRT", actual["xla"])
    self.assertEqual("some xla_flags", actual["xla_flags"])
    self.assertEqual("openxla", actual["dynamo"])
    self.assertEqual("train", actual["test"])
    self.assertEqual("123", actual["batch_size"])


if __name__ == '__main__':
  unittest.main()
