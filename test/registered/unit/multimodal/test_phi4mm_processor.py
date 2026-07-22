import unittest

from sglang.srt.multimodal.processors.phi4mm import Phi4MMProcessorAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _KeywordOnlyProcessor:
    def __init__(self):
        self.tokenizer = object()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [1, 2]}


class TestPhi4MMProcessorAdapter(CustomTestCase):
    def test_preserves_tokenizer_and_keyword_delegation(self):
        processor = _KeywordOnlyProcessor()
        adapter = Phi4MMProcessorAdapter(processor)

        self.assertIs(adapter.tokenizer, processor.tokenizer)
        self.assertEqual(adapter(text="hello"), {"input_ids": [1, 2]})
        self.assertEqual(processor.calls, [{"text": "hello"}])


if __name__ == "__main__":
    unittest.main()
