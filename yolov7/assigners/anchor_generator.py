import torch


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,
                     device='cpu', is_eval=False, mode='af'):
    """根据特征图来生成锚框"""
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None

    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor_point = torch.stack(
                [shift_x, shift_y], dim=1
            ).to(torch.float32)

            if mode == 'af':  # 无锚框anchor-free模式
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                    torch.full(size=(h*w, 1), fill_value=stride,
                               dtype=torch.float, device=device)
                )
            elif mode == 'ab':  # 有锚框anchor-based模式
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                stride_tensor.append(
                    torch.full(
                        size=(h*2, 1), fill_value=stride, dtype=torch.float, device=device
                    ).repeat(3, 1)
                )
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ], dim=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], dim=-1
            ).clone().to(feats[0].dtype)

            if mode == 'af':  # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab':  # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3, 1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
            num_anchors_list.append(len(anchors[-1]))

            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype
                )
            )

        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        """
        输出内容的解析：
        1. anchors (Tensor): shape=[8400, 4], 表示总共8400个生成框，每个生成框的预设大小的四角坐标
            信息
        2. anchor_points (Tensor): shape=[8400, 2], 表示总共8400个生成框，每个生成框的中心坐标点位
        3. stride_tensor (Tensor): shape=[8400, 1], 表示总共8400个生成框，每个特征点所代表的像素大
            小；比如第一层级的特征图最终为 80 * 80，那么相对于原来的图片大小 640 * 640 来说，每个特征点
            代表的像素大小为 640 / 80 = 8
        4. num_anchors_list (list): [6400, 1600, 400], 表示一共三个层级，每个层级的特征点数量分别
            是 6400， 1600， 400 个特征框
        """
        return anchors, anchor_points, num_anchors_list, stride_tensor


if __name__ == '__main__':
    feats = [torch.rand(16, 128, 80, 80),
             torch.rand(16, 256, 40, 40),
             torch.rand(16, 512, 20, 20)]
    fpn_strides = [8, 16, 32]
    grid_cell_size = 5.0
    grid_cell_offset = 0.5
    anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
        feats, fpn_strides, grid_cell_size, grid_cell_offset, device='cpu'
    )
