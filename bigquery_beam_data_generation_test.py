"""Tests for tfx.examples.tfjs_next_page_prediction.biquery_beam_data_generation."""
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import tensorflow as tf
from tfx.examples.tfjs_next_page_prediction import bigquery_beam_data_generation


class BiqueryBeamDataGenerationTest(tf.test.TestCase):

  def testExampleGeneration(self):
    expected_ga_session = {
        'hits': [
            {
                'hitNumber': 2,
                'time': 1,
                'page': {
                    'pagePath': 'page2?mypage'
                }
            },
            {
                'hitNumber': 1,
                'time': 0,
                'page': {
                    'pagePath': 'page1'
                }
            },
            {
                'hitNumber': 3,
                'time': 2,
                'page': {
                    'pagePath': 'page3'
                }
            },
            {
                'hitNumber': 4,
                'time': 2,
                'page': {
                    'pagePath': 'page3'
                }
            },
        ]
    }

    expected_training_examples = [
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'cur_page':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b'page1'])),
                    'session_index':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[0])),
                    'label':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b'page2'])),
                })),
        tf.train.Example(
            features=tf.train.Features(
                feature={
                    'cur_page':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b'page2'])),
                    'session_index':
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[1])),
                    'label':
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b'page3'])),
                }))
    ]
    with TestPipeline() as p:
      run_result = (
          p | beam.Create([expected_ga_session])
          | beam.ParDo(bigquery_beam_data_generation.ExampleGeneratingDoFn()))
      assert_that(run_result, equal_to(expected_training_examples))


if __name__ == '__main__':
  tf.test.main()