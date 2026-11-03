"""Isolated scratch coordinator check: mocked selectors and agents, no runtime writes."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class ScratchPipelineVisualInputsTests(unittest.TestCase):
    def test_native_and_critic_sequences_use_new_selector_without_mutating_active_modules(self):
        worker=load('scratch_pipeline_visual_inputs',HERE/'pipeline.py')
        selector=load('scratch_selector_visual_inputs',HERE/'repair_visual_inputs.py')
        fixtures=load('scratch_existing_visual_memory_tests',HERE.parents[1]/'tests/test_visual_history_memory.py')
        selected={'status':'available','images':[{'id':f'analytic-{i}','path':f'/analytic/image-{i}.png','sha256':str(i)*64} for i in range(5)]}
        with patch.dict(sys.modules,{'avatar_assets.repair_visual_inputs':selector}), \
             patch.object(fixtures,'pipeline',worker), \
             patch.object(selector,'from_continuation',return_value=selected) as inherited, \
             patch.object(selector,'from_current_review',return_value=selected) as current:
            fixtures.VisualHistoryMemoryTests()._sequence(True)
            fixtures.VisualHistoryMemoryTests()._sequence(False)
            self.assertEqual(inherited.call_count,2);self.assertEqual(current.call_count,2)

if __name__=='__main__':unittest.main()
