# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3


from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
import matplotlib.cm
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
from skimage.feature import canny
from skimage.morphology import binary_dilation

import torch
from PIL import Image, ImageDraw, ImageFont
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid
from utils import im_util, logging_util, structs, torch_helpers

matplotlib.use("agg")

logger = logging_util.get_logger(__name__)

DEFAULT_HAND_COLORS = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
DEFAULT_FG_OPACITY = 0.5
DEFAULT_BG_OPACITY = 0.7
DEFAULT_FONT_SIZE = 15
DEFAULT_TXT_OFFSET = (2, 1)


def normalize_data(img):
    return (img - img.min()) / (img.max() - img.min())


def get_colormap(num_colors, cmap=cv2.COLORMAP_TURBO):
    palette = np.linspace(0, 255, num_colors, dtype=np.uint8).reshape(1, num_colors, 1)
    palette = cv2.applyColorMap(palette, cmap)
    palette = cv2.cvtColor(palette, cv2.COLOR_BGR2RGB).squeeze(0)
    palette = normalize_data(palette)
    return palette.tolist()


def plot_images(
    imgs,
    titles=None,
    cmaps="gray",
    dpi: int = 100,
    pad: float = 0.0,
    adaptive: bool = True,
) -> None:

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    im_width = imgs[0].shape[1]
    im_height = imgs[0].shape[0]

    if n > 1:
        for im_id in range(len(imgs[1:])):
            if imgs[im_id].shape[1] != im_width or imgs[im_id].shape[0] != im_height:
                raise ValueError("Images must be of the same size.")

    figsize = (len(imgs) * im_width / dpi, im_height / dpi)
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors="lime", ps=4) -> None:

    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(ps, list):
        ps = [ps] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c, p in zip(axes, kpts, colors, ps):
        a.scatter(k[:, 0], k[:, 1], c=c, s=p, linewidths=0)


def plot_matches(
    kpts0,
    kpts1,
    color: Optional[List[float]] = None,
    lw: float = 1.5,
    ps: int = 4,
    indices: Tuple[int, int] = (0, 1),
    a=1.0,
    w: int = 640,
    h: int = 480,
) -> None:

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    mask = np.logical_and(
        np.logical_and(0 <= kpts1[:, 0], kpts1[:, 0] < w),
        np.logical_and(0 <= kpts1[:, 1], kpts1[:, 1] < h),
    )
    kpts1 = kpts1[mask]
    kpts0 = kpts0[mask]

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)  # pyre-ignore
    else:
        color = np.array(color)[mask].tolist()

    if type(a) is not list:
        a = [a for i in range(len(kpts0))]

    if lw > 0:
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                zorder=1,
                transform=fig.transFigure,
                c=color[i],
                linewidth=lw,
                alpha=a[i],
            )
            for i in range(len(kpts0))
        ]

    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_boundingbox(box) -> None:

    x1, y1, x2, y2 = box
    crop_width, crop_height = x2 - x1, y2 - y1
    rect = patches.Rectangle(
        (x1, y1),
        crop_width,
        crop_height,
        linewidth=1,
        edgecolor="white",
        facecolor="none",
    )
    ax = plt.gca()
    ax.add_patch(rect)


def plot_curve(x, y, xlabel: str = "x axis", ylabel: str = "y axis", title: str = "plot") -> None:

    plt.figure()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.plot(x, y)


def plot_histogram(
    hist_values,
    n_bins,
    value,
    im_width: int = 630,
    im_height: int = 476,
    dpi: int = 100,
    pad: float = 0.0,
    colors: Optional[List[int]] = None,
    amb: bool = False,
) -> None:

    figsize = (im_width / dpi, im_height / dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    n, bins, patches = ax.hist(hist_values, bins=n_bins)
    ax.set_xlabel(value)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {value}")

    if colors is not None:
        if len(colors) == len(bins):
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[i])
        elif amb:
            prev = 0
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[prev])
                if n[i] != 0:
                    prev += 1
        else:
            prev = 0
            for i, patch in enumerate(patches):
                patch.set_facecolor(colors[prev])
                if n[i] != 0:
                    prev += int(n[i])

    fig.tight_layout(pad=pad)


def plot_bar(
    features,
    importances,
    indices,
    x_label: str = "Relative Importance",
    title: str = "Feature importances using MDI",
    figsize: Tuple[int, int] = (20, 20),
) -> None:
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=7)
    plt.xlabel(x_label)


def add_contour_overlay(
    img,
    mask_img,
    color: Optional[Tuple] = (255, 255, 255),
    dilate_iterations: Optional[int] = 1,
):

    img_t = torch.as_tensor(mask_img)
    mask = torch.zeros_like(img_t)
    mask[img_t > 0] = 255
    mask = torch.max(mask, dim=-1)[0]
    mask_bool = mask.numpy().astype(bool)

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)[:, :, None]
    mask_rgb = np.concatenate((mask_uint8, mask_uint8, mask_uint8), axis=-1)

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)

    kernel = np.ones((3, 3), np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)

    img_contour = np.copy(img)
    img_contour[canny > 0] = color

    return img_contour


def add_text(
    idx,
    text,
    pos: Tuple[float, float] = (0.01, 0.99),
    fs: int = 12,  # 15,
    color: str = "w",
    lcolor: str = "k",
    lwidth: int = 2,
    ha: str = "left",
    va: str = "top",
) -> None:
    fig = plt.gcf()
    ax = fig.axes[idx]

    zorder = ax.get_zorder()
    if len(fig.axes) > 1:
        zorder = max(zorder, fig.axes[-1].get_zorder())
        fig.axes[-1].set_axisbelow(True)

    t = ax.text(
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=ax.transAxes,
        zorder=zorder + 5,
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )
        t.set


def save_plot_to_ndarray():
    fig = plt.gcf()
    fig.canvas.draw()
    # requires matplotlib 3.9.0 (<3.10)
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.clf()
    plt.cla()
    plt.close(fig)

    return data


def draw_contour(img_PIL: Image.Image, mask: np.ndarray, color: structs.Color, to_pil: bool = True):

    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    img = np.array(img_PIL)
    img[edge, :] = color
    if to_pil:
        return Image.fromarray(img)
    else:
        return img


def blend_images(ims: Iterable[npt.NDArray], weights: Iterable[float]) -> npt.NDArray:

    im_shape = None
    for im in ims:
        if im_shape is None:
            im_shape = im.shape
        elif im_shape != im.shape:
            raise ValueError(f"Images must be of the same size ({im_shape} vs {im.shape}).")
        if im.dtype != np.uint8:
            raise ValueError(f"Images must be of type uint8 ({im.dtype}).")

    blend = np.zeros(list(im_shape), dtype=np.float32)
    for im, weight in zip(ims, weights):
        blend += weight * im.astype(np.float32)
    blend[blend > 255] = 255
    return blend.astype(np.uint8)


def vis_masks(
    base_image: np.ndarray,
    mask: List[np.ndarray],
    color: List[Tuple[float, float, float]],
):

    color_mask = np.zeros_like(base_image, dtype=np.uint8)
    color_mask[mask > 0] = color

    out_image = blend_images([base_image.copy().astype(np.uint8), color_mask], [0.5, 0.5])
    return out_image


def vis_masks_on_rgbs(
    rgbs: torch.Tensor,
    masks: torch.Tensor,
    color: Tuple[float, float, float] = (255, 0, 0),
) -> torch.Tensor:

    device = rgbs.device
    rgbs_numpy = torch_helpers.tensor_to_array(rgbs)
    masks_numpy = torch_helpers.tensor_to_array(masks)

    b, h, w = masks_numpy.shape
    overlay_rgbs = np.zeros((b, h, w, 3), dtype=np.uint8)
    for i, (rgb, mask) in enumerate(zip(rgbs_numpy, masks_numpy)):
        overlay_rgbs[i] = vis_masks(
            base_image=im_util.chw_to_hwc(rgb) * 255,
            mask=mask,
            color=color,
        )
    overlay_rgb = torch.from_numpy(overlay_rgbs).to(device) / 255.0
    return overlay_rgb


def add_border_numpy(
    image: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    border_size: int = 5,
) -> np.ndarray:

    image[:border_size, :] = color
    image[-border_size:, :] = color
    image[:, :border_size] = color
    image[:, -border_size:] = color
    return image


def rgb_from_error_map(error_map: np.ndarray, max_norm: Optional[float] = None) -> np.ndarray:

    if max_norm is None:
        max_norm = error_map.max()
    normalized_error_map = np.clip(error_map, 0, max_norm) / max_norm

    rgb_image = plt.cm.plasma(normalized_error_map)[:, :, :3] * 255
    return rgb_image


def grays_from_rgbs(rgbs: torch.Tensor) -> torch.Tensor:

    rgbs_numpy = torch_helpers.tensor_to_array(rgbs)
    grays_numpy = []
    for i in range(rgbs_numpy.shape[0]):
        rgb = np.uint8(im_util.chw_to_hwc(rgbs_numpy[i]) * 255.0)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grays_numpy.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    grays_numpy = np.stack(grays_numpy)
    grays = torch.from_numpy(grays_numpy).float() / 255.0
    grays = grays.permute(0, 3, 1, 2)
    return grays


def wrap_flows(
    rgbs_source: torch.Tensor,
    flows: torch.Tensor,
    rgbs_target: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    height, width = rgbs_source.shape[2], rgbs_source.shape[3]
    batch_size = rgbs_source.shape[0]

    target_pixels = create_meshgrid(height, width, normalized_coordinates=False)

    target_pixels = target_pixels.repeat(batch_size, 1, 1, 1)

    source_pixels = target_pixels - flows

    wrapped_rgbs = remap(
        rgbs_source,
        source_pixels[..., 0],
        source_pixels[..., 1],
        mode="bilinear",
    )

    if rgbs_target is not None:
        wrapped_rgbs[wrapped_rgbs == 0] = rgbs_target[wrapped_rgbs == 0]
    return wrapped_rgbs


class VideoWriter(object):
    pass


class TrackingVisualizer(object):
    def __init__(
        self,
        save_dir: str,
        fps: int = 10,
    ) -> None:
        super(TrackingVisualizer, self).__init__()
        self.save_dir: str = save_dir
        self.fps: int = fps

        self.last_step_writer = VideoWriter(f"{self.save_dir}/_last_step.mp4", fps=10)
        logger.info(f"Visualizer initialized: {self.save_dir}")

    def close(self) -> None:
        self.last_step_writer.close()
        logger.info(f"Tracking video (last step) saved at {self.last_step_writer._path}")

    def write(
        self,
        vis_grids: Union[np.ndarray, List[np.ndarray]],
    ) -> None:
        if isinstance(vis_grids, np.ndarray):

            vis_grids = [vis_grids]
        self.last_step_writer.write(vis_grids[-1])


def write_text_on_image(
    im: npt.NDArray,
    txt_list: Sequence[Dict[str, Any]],
    loc: Tuple[int, int] = DEFAULT_TXT_OFFSET,
    color: structs.Color = (1.0, 1.0, 1.0),
    size: int = 20,
) -> npt.NDArray:

    im_pil = Image.fromarray(im)

    try:
        root_dir = Path(__file__).parent.parent
        font_path = root_dir / "external/bop_toolkit/bop_toolkit_lib/droid_sans_mono.ttf"
        font = ImageFont.truetype(str(font_path), size)
    except IOError:
        logger.info("Warning: Loading a fallback font.")
        font = ImageFont.load_default()

    im_size = (im.shape[1], im.shape[0])
    loc = tuple(im_util.clip_2d_point(torch.as_tensor(loc), torch.as_tensor(im_size)))

    draw = ImageDraw.Draw(im_pil)
    for info in txt_list:
        txt = ""
        if "name" in info:
            txt += info["name"]
        if "val" in info:
            if "fmt" in info:
                val_tpl = "{" + info["fmt"] + "}"
            elif type(info["val"]) == float:  # noqa: E721
                val_tpl = "{:.3f}"
            else:
                val_tpl = "{}"
            if txt != "":
                txt += ": "
            txt += val_tpl.format(info["val"])
        draw.text(
            xy=loc,
            text=txt,
            fill=tuple([int(255 * c) for c in color]),
            font=font,
        )
        text_bbox = font.getbbox("X")
        text_height = text_bbox[3] - text_bbox[1]
        loc = (loc[0], loc[1] + int(1.3 * text_height))
    del draw

    return np.array(im_pil)


def build_grid(
    tiles: Sequence[npt.NDArray],
    tile_size: Optional[Tuple[int, int]] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    max_height: int = 800,
    tile_pad: int = 5,
    interpolation: Optional[Any] = None,
) -> npt.NDArray:

    shape_len = len(tiles[0].shape)
    assert shape_len in [2, 3], f"Invalid tile shape: {tiles[0].shape}"
    if tile_size is None:
        height = min(max_height, max(tile.shape[0] for tile in tiles))
        resized_tiles = []
        for tile in tiles:
            tile_size = (int(tile.shape[1] * height / tile.shape[0]), height)
            tile = im_util.resize_image(tile, tile_size, interpolation=interpolation)
            if shape_len == 3:
                pad_width = ((tile_pad, tile_pad), (tile_pad, tile_pad), (0, 0))
            elif shape_len == 2:
                pad_width = ((tile_pad, tile_pad), (tile_pad, tile_pad))
            tile = np.pad(tile, pad_width=pad_width, mode="constant", constant_values=255)
            resized_tiles.append(tile)
        grid = np.hstack(resized_tiles)

    else:
        if grid_rows is None or grid_cols is None:
            grid_rows = int(np.sqrt(len(tiles)))
            grid_cols = int(np.ceil(len(tiles) / grid_rows))
        if shape_len == 3:
            grid = np.zeros((grid_rows * tile_size[1], grid_cols * tile_size[0], 3), np.uint8)
        elif shape_len == 2:
            grid = np.zeros((grid_rows * tile_size[1], grid_cols * tile_size[0]), np.uint8)
        w, h = tile_size
        for tile_id, tile in enumerate(tiles):
            if tile_size != (tile.shape[1], tile.shape[0]):
                tile = im_util.resize_image(tile, tile_size, interpolation=interpolation)
            yy = int(tile_id / grid_cols)
            xx = tile_id % grid_cols
            grid[(yy * h) : ((yy + 1) * h), (xx * w) : ((xx + 1) * w)] = tile

    return grid


def add_text_and_merge_tiles(
    tiles: List[npt.NDArray],
    texts: List[str],
    text_size: int = 12,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
) -> List[npt.NDArray]:

    img_shape = tiles[0].shape

    batch_size, height, width = img_shape[:3]
    text_size = int(text_size * width / 224)

    for idx, tile in enumerate(tiles):
        assert tile.shape == img_shape, f"Image {idx} has {tile.shape} != {img_shape}."

    merged_tiles = []
    for sample_id in range(batch_size):
        merged_tile = []
        for tile_id in range(len(tiles)):
            tile = np.asarray(tiles[tile_id][sample_id] * 255.0, dtype=np.uint8)
            tile = write_text_on_image(
                tile,
                [{"name": texts[tile_id]}],
                size=text_size,
            )
            merged_tile.append(tile)

        merged_tile = build_grid(
            tiles=merged_tile,
            tile_size=(width, height),
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        merged_tiles.append(merged_tile)
    return merged_tiles
