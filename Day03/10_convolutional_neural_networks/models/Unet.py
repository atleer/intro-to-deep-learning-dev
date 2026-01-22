'''
Copyright 2020 Neuromatch Academy

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import torch.nn as nn

def convbatchrelu(in_channels, out_channels, sz):
    """
    Creates a sequential block of Conv2d -> BatchNorm2d -> ReLU.
    
    This is a common building block in deep neural networks that applies:
    1. 2D convolution with same padding to preserve spatial dimensions
    2. Batch normalization for training stability
    3. ReLU activation for non-linearity
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        sz (int): Kernel size for the convolution
        
    Returns:
        nn.Sequential: Sequential container with Conv2d, BatchNorm2d, and ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


class convdown(nn.Module):
    """
    Downsampling convolutional block that applies two consecutive convbatchrelu blocks.
    
    This is used in the encoder (downsampling) path of the U-Net to extract features
    while maintaining the same spatial resolution within each level.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Initialize the convdown block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolutions
        """
        super().__init__()
        self.conv = nn.Sequential()
        
        # TODO: I don't think the for loop is necessary. Remove and test
        # Add two convbatchrelu blocks
        for t in range(2):
            if t == 0:
                # First conv: input_channels -> output_channels
                self.conv.add_module('conv_%d'%t,
                                   convbatchrelu(in_channels,
                                               out_channels,
                                               kernel_size))
            else:
                # Second conv: output_channels -> output_channels
                self.conv.add_module('conv_%d'%t,
                                   convbatchrelu(out_channels,
                                               out_channels,
                                               kernel_size))

    def forward(self, x):
        """
        Forward pass through both convolution blocks.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after two convolution blocks
        """
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class downsample(nn.Module):
    """
    Encoder path of the U-Net that progressively downsamples the input.
    
    Creates multiple levels of feature extraction, where each level:
    1. Applies convdown to extract features
    2. Uses max pooling to reduce spatial dimensions for the next level
    """
    
    def __init__(self, nbase, kernel_size):
        """
        Initialize the downsampling path.
        
        Args:
            nbase (list): List of channel numbers for each level [input_ch, level1_ch, level2_ch, ...]
            kernel_size (int): Kernel size for all convolutions
        """
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)  # Reduces spatial dimensions by factor of 2
        
        # Create convdown blocks for each level transition
        for n in range(len(nbase) - 1):
            self.down.add_module('conv_down_%d'%n,
                               convdown(nbase[n],
                                      nbase[n + 1],
                                      kernel_size))

    def forward(self, x):
        """
        Forward pass through the encoder path.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            list: List of feature maps from each level (for skip connections)
        """
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                # Apply max pooling to previous level's output
                y = self.maxpool(xd[n - 1])
            else:
                # First level uses original input
                y = x
            # Apply convdown block and store result
            xd.append(self.down[n](y))
        return xd


class convup(nn.Module):
    """
    Upsampling convolutional block for the decoder path.
    
    Combines upsampled features with skip connection features from the encoder
    and applies two convolution blocks.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Initialize the convup block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolutions
        """
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', convbatchrelu(in_channels,
                                                   out_channels,
                                                   kernel_size))
        self.conv.add_module('conv_1', convbatchrelu(out_channels,
                                                   out_channels,
                                                   kernel_size))

    def forward(self, x, y):
        """
        Forward pass combining upsampled features with skip connections.
        
        Args:
            x (torch.Tensor): Upsampled features from previous decoder level
            y (torch.Tensor): Skip connection features from corresponding encoder level
            
        Returns:
            torch.Tensor: Combined and processed features
        """
        x = self.conv[0](x)
        x = self.conv[1](x + y)  # Skip connection: element-wise addition
        return x


class upsample(nn.Module):
    """
    Decoder path of the U-Net that progressively upsamples features.
    
    Uses skip connections from the encoder path to recover spatial information
    lost during downsampling.
    """
    
    def __init__(self, nbase, kernel_size):
        """
        Initialize the upsampling path.
        
        Args:
            nbase (list): List of channel numbers for each level (reverse order of encoder)
            kernel_size (int): Kernel size for all convolutions
        """
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')  # 2x spatial upsampling
        self.up = nn.Sequential()
        
        # Create convup blocks for each level (reverse order)
        for n in range(len(nbase) - 1, 0, -1):
            self.up.add_module('conv_up_%d'%(n - 1),
                             convup(nbase[n], nbase[n - 1], kernel_size))

    def forward(self, xd):
        """
        Forward pass through the decoder path.
        
        Args:
            xd (list): List of encoder feature maps (for skip connections)
            
        Returns:
            torch.Tensor: Final upsampled feature map
        """
        x = xd[-1]  # Start with deepest (most downsampled) features
        
        for n in range(0, len(self.up)):
            if n > 0:
                # Upsample spatial dimensions
                x = self.upsampling(x)
            # Combine with corresponding encoder features via skip connection
            x = self.up[n](x, xd[len(xd) - 1 - n])
        return x


class Unet(nn.Module):
    """
    Complete U-Net architecture for image segmentation.
    
    U-Net consists of:
    1. Encoder path (downsample): Extracts hierarchical features
    2. Decoder path (upsample): Reconstructs spatial resolution using skip connections
    3. Final output layer: Produces segmentation map
    
    The skip connections between encoder and decoder help recover fine-grained
    spatial information lost during downsampling.
    """
    
    def __init__(self, nbase, nout, kernel_size):
        """
        Initialize the U-Net model.
        
        Args:
            nbase (list): Channel configuration [input_channels, level1, level2, ..., bottleneck]
            nout (int): Number of output classes/channels
            kernel_size (int): Kernel size for all convolutions
            
        Example:
            nbase = [1, 64, 128, 256, 512] for a 4-level U-Net with 1 input channel
            nout = 2 for binary segmentation
        """
        super(Unet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.kernel_size = kernel_size
        
        # Encoder path
        self.downsample = downsample(nbase, kernel_size)
        
        # Decoder path configuration (skip the input level, add bottleneck level)
        nbaseup = nbase[1:]  # Remove input channels
        nbaseup.append(nbase[-1])  # Add bottleneck channels
        self.upsample = upsample(nbaseup, kernel_size)
        
        # Final output convolution to produce segmentation map
        self.output = nn.Conv2d(nbase[1], self.nout, kernel_size,
                              padding=kernel_size//2)

    def forward(self, data):
        """
        Forward pass through the complete U-Net.
        
        Args:
            data (torch.Tensor): Input image tensor [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Segmentation output [batch, nout, height, width]
        """
        # Encoder: extract multi-scale features
        T0 = self.downsample(data)
        
        # Decoder: reconstruct with skip connections
        T0 = self.upsample(T0)
        
        # Final classification layer
        T0 = self.output(T0)
        return T0

    def save_model(self, filename):
        """
        Save model parameters to file.
        
        Args:
            filename (str): Path to save the model state dictionary
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        """
        Load model parameters from file.
        
        Args:
            filename (str): Path to the saved model state dictionary
            cpu (bool): If True, load model for CPU inference
            
        Note:
            The CPU loading branch has a bug - it references 'self.concatenation' 
            which doesn't exist in __init__. This should be removed or fixed.
        """
        if not cpu:
            self.load_state_dict(torch.load(filename))
        else:
            # BUG: This reinitializes the model but references non-existent self.concatenation
            self.__init__(self.nbase,
                        self.nout,
                        self.kernel_size,
                        self.concatenation)  # This line will cause an AttributeError

            self.load_state_dict(torch.load(filename,
                                          map_location=torch.device('cpu')))

# Example usage:
# model = Unet(nbase=[3, 64, 128, 256, 512], nout=1, kernel_size=3)
# This creates a U-Net with:
# - 3 input channels (RGB image)
# - 4 encoder/decoder levels with [64, 128, 256, 512] channels
# - 1 output channel (binary segmentation)
# - 3x3 convolution kernels throughout