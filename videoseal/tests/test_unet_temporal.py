"""
Tests for the UNetMsg temporal pooling functionality.
Run with:
    python -m videoseal.tests.test_unet_temporal
"""

import torch
import unittest
import numpy as np
from typing import Tuple, List
from videoseal.modules.msg_processor import MsgProcessor
from videoseal.modules.unet import UNetMsg


class TestUNetTemporal(unittest.TestCase):
    def setUp(self):
        """Setup common test parameters"""
        # Common parameters for all tests
        self.nbits = 8
        self.hidden_size = 16
        self.in_channels = 3
        self.out_channels = 3
        self.z_channels = 32
        self.z_channels_mults = (1, 2, 4, 8)  # 4 depths (0 to 3)
        
        # Create message processor
        self.msg_processor = MsgProcessor(
            nbits=self.nbits,
            hidden_size=self.hidden_size,
            msg_processor_type="binary+concat"
        )
        
        # Test data dimensions
        self.batch_size = 4
        self.height = 64
        self.width = 64
        
        # Generate test inputs
        self.imgs = torch.randn(self.batch_size, self.in_channels, self.height, self.width)
        self.msgs = self.msg_processor.get_random_msg(self.batch_size)
        
    def _create_model(self, time_pooling: bool, time_pooling_depth: int, time_pooling_kernel_size: int = 2) -> UNetMsg:
        """Helper to create model with specific temporal pooling settings"""
        return UNetMsg(
            msg_processor=self.msg_processor,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            z_channels=self.z_channels,
            num_blocks=2,
            activation="relu",
            normalization="batch",
            z_channels_mults=self.z_channels_mults,
            time_pooling=time_pooling,
            time_pooling_depth=time_pooling_depth,
            time_pooling_kernel_size=time_pooling_kernel_size
        )
    
    def test_no_temporal_pooling(self):
        """Test UNetMsg without temporal pooling"""
        model = self._create_model(time_pooling=False, time_pooling_depth=0)
        model.eval()
        
        with torch.no_grad():
            output = model(self.imgs, self.msgs)
        
        # Output should match input batch size
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.out_channels)
        self.assertEqual(output.shape[2:], (self.height, self.width))
    
    def test_all_temporal_depths(self):
        """Test UNetMsg with temporal pooling at all possible depths"""
        max_depth = len(self.z_channels_mults) - 2  # Maximum valid depth
        
        for depth in range(max_depth + 1):
            with self.subTest(f"Testing time_pooling_depth={depth}"):
                model = self._create_model(time_pooling=True, time_pooling_depth=depth)
                model.eval()
                
                with torch.no_grad():
                    output = model(self.imgs, self.msgs)
                
                # Output batch size should match input batch size
                self.assertEqual(output.shape[0], self.batch_size)
                self.assertEqual(output.shape[1], self.out_channels)
                self.assertEqual(output.shape[2:], (self.height, self.width))
    
    def test_varying_kernel_sizes(self):
        """Test different temporal pooling kernel sizes"""
        kernel_sizes = [2, 4]
        depth = 1  # Use a fixed depth
        
        for kernel_size in kernel_sizes:
            with self.subTest(f"Testing kernel_size={kernel_size}"):
                # Adjust batch size to be divisible by kernel_size
                adjusted_batch = kernel_size * 3
                test_imgs = torch.randn(adjusted_batch, self.in_channels, self.height, self.width)
                test_msgs = self.msg_processor.get_random_msg(adjusted_batch)
                
                model = self._create_model(
                    time_pooling=True, 
                    time_pooling_depth=depth,
                    time_pooling_kernel_size=kernel_size
                )
                model.eval()
                
                with torch.no_grad():
                    output = model(test_imgs, test_msgs)
                
                # Output should maintain original batch size
                self.assertEqual(output.shape[0], adjusted_batch)
    
    def test_model_consistency(self):
        """Test that model produces consistent outputs for same inputs"""
        depth = 1
        model = self._create_model(time_pooling=True, time_pooling_depth=depth)
        model.eval()
        
        with torch.no_grad():
            output1 = model(self.imgs, self.msgs)
            output2 = model(self.imgs, self.msgs)
        
        # Outputs should be identical for same inputs
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_asymmetric_batch_size(self):
        """Test with batch sizes not divisible by kernel size"""
        kernel_size = 3
        depth = 1
        odd_batch = 5  # Not divisible by kernel_size
        
        test_imgs = torch.randn(odd_batch, self.in_channels, self.height, self.width)
        test_msgs = self.msg_processor.get_random_msg(odd_batch)
        
        model = self._create_model(
            time_pooling=True, 
            time_pooling_depth=depth,
            time_pooling_kernel_size=kernel_size
        )
        model.eval()
        
        with torch.no_grad():
            # Should handle this case without errors
            output = model(test_imgs, test_msgs)
        
        # Output should match input batch size
        self.assertEqual(output.shape[0], odd_batch)


def run_temporal_pooling_demo():
    """Run a visual demo of temporal pooling at different depths"""
    # Configuration
    nbits = 8
    hidden_size = 16
    in_channels = 3
    out_channels = 3
    z_channels = 32
    z_channels_mults = (1, 2, 4, 8)  # 4 depths
    
    msg_processor = MsgProcessor(
        nbits=nbits,
        hidden_size=hidden_size,
        msg_processor_type="binary+concat"
    )
    
    # Create test inputs with easily identifiable patterns
    batch_size = 8
    height = 64
    width = 64
    
    # Create a gradient pattern for visualization
    test_imgs = torch.zeros(batch_size, in_channels, height, width)
    for i in range(batch_size):
        # Create a unique pattern for each batch item
        intensity = (i + 1) / batch_size
        test_imgs[i, 0] = intensity  # Red channel gets stronger with batch index
        test_imgs[i, 1] = 1.0 - intensity  # Green channel gets weaker with batch index
    
    msgs = msg_processor.get_random_msg(batch_size)
    
    # Test temporal pooling at each depth
    max_depth = len(z_channels_mults) - 2
    results = {}
    
    print(f"\nRunning temporal pooling demo with {batch_size} frames...")
    for depth in range(max_depth + 1):
        model = UNetMsg(
            msg_processor=msg_processor,
            in_channels=in_channels,
            out_channels=out_channels,
            z_channels=z_channels,
            num_blocks=2,
            activation="relu",
            normalization="batch",
            z_channels_mults=z_channels_mults,
            time_pooling=True,
            time_pooling_depth=depth,
            time_pooling_kernel_size=2
        )
        model.eval()
        
        with torch.no_grad():
            output = model(test_imgs, msgs)
        
        # Store statistics for analysis
        results[depth] = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'shape': output.shape
        }
        
        print(f"Depth {depth}: Output shape {output.shape}, Mean: {results[depth]['mean']:.4f}, Std: {results[depth]['std']:.4f}")
    
    print("\nTemporal pooling demo completed.")
    return results


if __name__ == "__main__":
    print("Running UNetMsg temporal pooling tests...")
    
    # Run the visual demo
    demo_results = run_temporal_pooling_demo()
    
    # Run all tests
    unittest.main()
