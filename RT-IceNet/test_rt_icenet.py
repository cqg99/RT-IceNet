import unittest
import torch
from skfm import SKFM
from rt_icenet import RTIceNet


torch.set_num_threads(2)


class ManuscriptModelTests(unittest.TestCase):
    def test_skfm_uniform_weights_average_inputs(self):
        fusion = SKFM(16)
        for parameter in fusion.parameters():
            torch.nn.init.zeros_(parameter)
        x = torch.randn(2, 16, 9, 11, requires_grad=True)
        y = torch.randn_like(x, requires_grad=True)
        actual = fusion(x, y)
        torch.testing.assert_close(actual, (x + y) / 2)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 0.5))
        torch.testing.assert_close(y.grad, torch.full_like(y, 0.5))

    def test_skfm_identical_inputs_are_preserved(self):
        fusion = SKFM(16)
        x = torch.randn(2, 16, 9, 11)
        torch.testing.assert_close(fusion(x, x), x)

    def test_output_shapes_and_backbone_called_once(self):
        for stride in (8, 16):
            model = RTIceNet(pretrained=False, downsample_factor=stride).eval()
            calls = []
            handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
            for height, width in ((64, 64), (65, 81)):
                with torch.no_grad():
                    output = model(torch.randn(1, 3, height, width))
                self.assertEqual(output.shape, (1, 4, height, width))
                self.assertTrue(torch.isfinite(output).all())
            handle.remove()
            self.assertEqual(len(calls), 2)

    def test_training_reaches_both_streams(self):
        model = RTIceNet(pretrained=False).train()
        output = model(torch.randn(2, 3, 64, 64))
        torch.nn.functional.cross_entropy(output, torch.randint(4, (2, 64, 64))).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_pafm_zero_gate_logits_give_equal_branch_weights(self):
        from pafm import PAFM
        fusion = PAFM(8)
        for parameter in fusion.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            fusion.conv.weight.copy_(torch.eye(8).reshape(8, 8, 1, 1))
        x = torch.ones(1, 8, 9, 11)
        y = torch.zeros_like(x)
        torch.testing.assert_close(fusion(x, y), x * 1.5)


if __name__ == '__main__':
    unittest.main()
