# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import PIL.Image
import scipy

import dnnlib
import dnnlib.tflib as tflib

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor

import warnings # mostly numpy warnings for me
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, outdir, minibatch_size=4):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    w_avg = Gs.get_var('dlatent_avg') # [component]
    Gs_syn_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'minibatch_size': minibatch_size
    }

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    os.makedirs(outdir, exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(f'{outdir}/grid.png')


# ----------------------------------------------------------------------------


def style_mixing_video(network_pkl,             # Path to pretrained model pkl file
                       src_seed,                # Seed of the source image style (row)
                       dst_seeds,               # Seeds of the destination image styles (columns)
                       col_styles,              # Styles to transfer from first row to first column
                       truncation_psi=1.0,      # Truncation trick
                       only_stylemix=False,     # True if user wishes to show only the style transferred result
                       outdir='out',
                       duration_sec=30.0,
                       smoothing_sec=3.0,
                       mp4_fps=30,
                       mp4_codec="libx264",
                       mp4_bitrate="16M",
                       minibatch_size=4):
    # Calculate the number of frames:
    num_frames = int(np.rint(duration_sec * mp4_fps))
    # Initialize TensorFlow
    tflib.init_tf()
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading networks from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)
    w_avg = Gs.get_var('dlatent_avg')  # [component]

    # Sanity check: styles are actually possible for generated image size
    max_style = int(2 * np.log2(Gs.output_shape[-1])) - 3
    assert max(col_styles) <= max_style, f"Maximum col-style allowed: {max_style}"

    Gs_syn_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'minibatch_size': minibatch_size
    }

    # First column latents
    print('Generating source W vectors...')
    src_shape = [num_frames] + Gs.input_shape[1:]
    src_z = np.random.RandomState(*src_seed).randn(*src_shape).astype(np.float32)  # [frames, src, component]
    src_z = scipy.ndimage.gaussian_filter(
        src_z,
        [smoothing_sec * mp4_fps] + [0] * (len(Gs.input_shape) - 1),
        mode="wrap"
    )
    src_z /= np.sqrt(np.mean(np.square(src_z)))
    # Map into the detangled latent space W and do truncation trick
    src_w = Gs.components.mapping.run(src_z, None)
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # First row latents
    print('Generating destination W vectors...')
    dst_z = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds])
    dst_w = Gs.components.mapping.run(dst_z, None)
    dst_w = w_avg + (dst_w - w_avg) * truncation_psi
    # Get the width and height of each image:
    _N, _C, H, W = Gs.output_shape

    # Generate ALL the source images:
    src_images = Gs.components.synthesis.run(src_w, **Gs_syn_kwargs)
    # Generate the column images:
    dst_images = Gs.components.synthesis.run(dst_w, **Gs_syn_kwargs)

    # If the user wishes to show both the source and destination images
    if not only_stylemix:
        print('Generating full video (including source and destination images)')
        # Generate our canvas where we will paste all the generated images:
        canvas = PIL.Image.new("RGB", (W * (len(dst_seeds) + 1), H * (len(src_seed) + 1)), 'black')

        for col, dst_image in enumerate(list(dst_images)):
            canvas.paste(PIL.Image.fromarray(dst_image, "RGB"), ((col + 1) * H, 0))

        # Paste them using an aux function for moviepy frame generation
        def make_frame(t):
            # Get the frame number according to time t:
            frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
            # We wish the image belonging to the frame at time t:
            src_image = src_images[frame_idx]
            # Paste it to the lower left:
            canvas.paste(PIL.Image.fromarray(src_image, "RGB"), (0, H))

            # Now, for each of the column images:
            for col, _ in enumerate(list(dst_images)):
                # Select the pertinent latent w column:
                w_col = np.stack([dst_w[col]])  # [18, 512] -> [1, 18, 512] for 1024x1024 images
                # Replace the values defined by col_styles:
                w_col[:, col_styles] = src_w[frame_idx, col_styles]
                # Generate these synthesized images:
                col_images = Gs.components.synthesis.run(w_col, **Gs_syn_kwargs)
                # Paste them in their respective spot:
                for row, image in enumerate(list(col_images)):
                    canvas.paste(
                        PIL.Image.fromarray(image, "RGB"),
                        ((col + 1) * H, (row + 1) * W),
                    )
            return np.array(canvas)
    # Else, show only the style-transferred images (this is nice for the 1x1 case)
    else:
        print('Generating only the style-transferred images')
        # Generate our canvas where we will paste all the generated images:
        canvas = PIL.Image.new("RGB", (W * len(dst_seeds), H * len(src_seed)), "white")

        def make_frame(t):
            # Get the frame number according to time t:
            frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
            # Now, for each of the column images:
            for col, _ in enumerate(list(dst_images)):
                # Select the pertinent latent w column:
                w_col = np.stack([dst_w[col]])  # [18, 512] -> [1, 18, 512]
                # Replace the values defined by col_styles:
                w_col[:, col_styles] = src_w[frame_idx, col_styles]
                # Generate these synthesized images:
                col_images = Gs.components.synthesis.run(w_col, **Gs_syn_kwargs)
                # Paste them in their respective spot:
                for row, image in enumerate(list(col_images)):
                    canvas.paste(
                        PIL.Image.fromarray(image, "RGB"),
                        (col * H, row * W),
                    )
            return np.array(canvas)
    # Generate video using make_frame:
    print('Generating style-mixed video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    grid_size = [len(dst_seeds), len(src_seed)]
    mp4 = "{}x{}-style-mixing.mp4".format(*grid_size)
    videoclip.write_videofile(os.path.join(outdir, mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)

#----------------------------------------------------------------------------

# My extended version of this helper function:
def _parse_num_range(s):
    '''
    Input:
        s (str): Comma separated string of numbers 'a,b,c', a range 'a-c', or
                 even a combination of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...'
    Output:
        nums (list): Ordered list of ascending ints in s, with repeating values
                     deleted (can be modified to not do either of this)
    '''
    # Sanity check 0:
    # In case there's a space between the numbers (impossible due to argparse,
    # but hey, I am that paranoid):
    s = s.replace(' ', '')
    # Split w.r.t comma
    str_list = s.split(',')
    nums = []
    for el in str_list:
        if '-' in el:
            # The range will be 'a-b', so we wish to find both a and b using re:
            range_re = re.compile(r'^(\d+)-(\d+)$')
            match = range_re.match(el)
            # We get the two numbers:
            a = int(match.group(1))
            b = int(match.group(2))
            # Sanity check 1: accept 'a-b' or 'b-a', with a<=b:
            if a <= b: r = [n for n in range(a, b + 1)]
            else: r = [n for n in range(b, a + 1)]
            # Use extend since r will also be an array:
            nums.extend(r)
        else:
            # It's a single number, so just append it:
            nums.append(int(el))
    # Sanity check 2: delete repeating numbers:
    nums = list(set(nums))
    # Return the numbers in ascending order:
    return sorted(nums)

#----------------------------------------------------------------------------

_examples = '''examples:

  python %(prog)s --outdir=out --trunc=1 --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate style mixing image matrix using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_grid = subparsers.add_parser('grid', help='Generate a grid of style-mixed images')
    parser_grid.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_grid.add_argument('--row-seeds', dest='row_seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_grid.add_argument('--col-seeds', dest='col_seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_grid.add_argument('--col-styles', dest='col_styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_grid.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_grid.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')
    parser_grid.set_defaults(func=style_mixing_example)

    parser_video = subparsers.add_parser('video', help='Generate style-mixing video (using lerp)')
    parser_video.add_argument('--network', help='Path to network pickle filename', dest='network_pkl', required=True)
    parser_video.add_argument('--row-seed', type=int, help='Random seed to use for image source row (content)', dest='src_seed', required=True)
    parser_video.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns (styles)', dest='dst_seeds', required=True)
    parser_video.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6', dest='col_styles')
    parser_video.add_argument('--only-stylemix', action='store_true', help='Add flag to only save the style-mixed images in the video', dest='only_stylemix')
    parser_video.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.7, dest='truncation_psi')
    parser_video.add_argument('--duration', type=float, help='Duration of video in seconds (default: %(default)s)', default=30, dest='duration_sec')
    parser_video.add_argument('--fps', type=int, help='FPS of generated video (default: %(default)s)', default=30, dest='mp4_fps')
    parser_video.add_argument('--outdir', help='Root directory for run results (default: %(default)s)', default='out', metavar='DIR')
    parser_video.set_defaults(func=style_mixing_video)

    args = parser.parse_args()
    kwargs = vars(args)
    submd = kwargs.pop('command')

    if submd is None:
        print('Error: missing subcommand. Re-run with --help for usage.')
        sys.exit(1)
    
    func = kwargs.pop('func')
    func(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
