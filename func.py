import torch


def normalize(image):
    """
    input: original image
    output: normalized image
    """
    return (image-image.min())/(image.max()-image.min()+1e-6)


def make_coord(shape, flatten=False):
    """
    input: shape: [D1, D2, ... , DN]
    output: coordinate index N × D1 × D2 × ... × DN
    """
    coord_seqs = []
    for _, n in enumerate(shape):
        seq = torch.arange(n).float()
        coord_seqs.append(seq)
    coord = torch.stack(torch.meshgrid(*coord_seqs), dim=0)
    if flatten:
        coord = coord.view(coord.shape[0], -1)
    return coord


def linear_interpolation(coord, img):
    coord_floor = coord.floor().long()  # 整数坐标
    coord_dif = coord - coord_floor  # 与整数坐标的偏移量
    dim = len(img.shape) - 1

    if dim == 2:
        x = coord_floor[0]
        y = coord_floor[1]
        u = coord_dif[0]
        v = coord_dif[1]
        # 坐标 → 强度值,并将坐标限制在[0,d1-1]与[0,d2-1],此例d1=174, d2=192
        get_img = lambda a, b: img[:, a.clamp(0, img.shape[-2] - 1), b.clamp(0, img.shape[-1] - 1)]
        return (
                (1 - u) * (1 - v) * get_img(x, y) +
                (1 - u) * v * get_img(x, y + 1) +
                u * (1 - v) * get_img(x + 1, y) +
                u * v * get_img(x + 1, y + 1)
        )
    elif dim == 3:
        x = coord_floor[0]
        y = coord_floor[1]
        z = coord_floor[2]
        u = coord_dif[0]
        v = coord_dif[1]
        t = coord_dif[2]
        get_img = lambda a, b, c: \
            img[: a.clamp(0, img.shape[-3] - 1), b.clamp(0, img.shape[-2] - 1), c.clamp(0, img.shape[-1] - 1)]
        return (
                (1 - u) * (1 - v) * (1 - t) * get_img(x, y, z) +
                (1 - u) * (1 - v) * t * get_img(x, y, z + 1) +
                (1 - u) * v * (1 - t) * get_img(x, y + 1, z) +
                (1 - u) * v * t * get_img(x, y + 1, z+1) +
                u * (1 - v) * (1 - t) * get_img(x + 1, y, z) +
                u * (1 - v) * t * get_img(x + 1, y, z + 1) +
                u * v * (1 - t) * get_img(x + 1, y + 1, z) +
                u * v * t * get_img(x+1, y + 1, z + 1)
        )


def SSD(input_tensor, target_tensor):
    return ((input_tensor - target_tensor) ** 2).mean()


def Affine_transformation(img_coord, A, b):
    new_img_coord = torch.mm(A, img_coord.view(img_coord.shape[0], -1)) + b
    return new_img_coord.view(img_coord.shape)


def cubic_Bspline(tensor):
    tensor = torch.abs(tensor)
    return (tensor < 1) * (2/3-tensor**2+(tensor**3)/2) + \
           ((tensor < 2) & (tensor >= 1))*(2-tensor)**3/6


def FFD_transformation(img_coord, mesh, h):
    shift = torch.zeros_like(img_coord).float()
    mesh_coord = (img_coord / h).floor().long()  # 浮动图像每个位置在网格中的左上角关键点坐标
    dim = img_coord.shape[0]  # 维度(dim=2)
    mesh_rel_coords = make_coord([4 for _ in range(dim)], flatten=True) - 1  # 定义了4 × 4的网格区域, 偏移量从[-1, -1]到[3, 3]共16个
    for mesh_rel_coord in mesh_rel_coords.T:
        if dim == 2:
            current_mesh_coord = mesh_coord + mesh_rel_coord.long().view(2, 1, 1)  # 目前所在的节点位置 = 像素点左上角节点坐标 + 当前偏移量
            B = cubic_Bspline(img_coord / h - current_mesh_coord)  # 以第一维x为例, 括号中即为 x / h - i - 当前偏移量(共两维, 后面要把两个维度乘起来)
            impact = B.prod(axis=0, keepdim=True)  # 影响系数1 × H × W, dim=0相乘以实现B(x/h-i-偏移量)B(y/h-j-偏移量)
            i = current_mesh_coord[0].clamp(0, mesh.shape[-2] - 1)  # 当前节点距像素点左上角x-dim偏移量
            j = current_mesh_coord[1].clamp(0, mesh.shape[-1] - 1)  # 当前节点距像素点左上角y-dim偏移量
            key_mesh = mesh[:, i, j]  # 关键点位移cij
            shift += impact * key_mesh  # 像素点位移
        elif dim == 3:
            current_mesh_coord = mesh_coord + mesh_rel_coord.long().view(3, 1, 1, 1)
            B = cubic_Bspline(img_coord / h - current_mesh_coord)
            impact = B.prod(0, keepdim=True)
            i = current_mesh_coord[0].clamp(0, mesh.shape[-3] - 1)
            j = current_mesh_coord[1].clamp(0, mesh.shape[-2] - 1)
            k = current_mesh_coord[1].clamp(0, mesh.shape[-1] - 1)
            key_mesh = mesh[:, i, j, k]
            shift += impact * key_mesh
    return img_coord + shift  # 像素点对应的新坐标, 后续经过插值得到对应像素值


def DFFD_transformation(img_coord, mesh, h):
    mesh = mesh.clamp(-0.4, 0.4)
    shift = torch.zeros_like(img_coord).float()
    mesh_coord = (img_coord / h).floor().long()  # 浮动图像每个位置在网格中的左上角关键点坐标
    dim = img_coord.shape[0]  # 维度(dim=2)
    mesh_rel_coords = make_coord([4 for _ in range(dim)], flatten=True) - 1  # 定义了4 × 4的网格区域, 偏移量从[-1, -1]到[3, 3]共16个
    for mesh_rel_coord in mesh_rel_coords.T:
        if dim == 2:
            current_mesh_coord = mesh_coord + mesh_rel_coord.long().view(2, 1, 1)  # 目前所在的节点位置 = 像素点左上角节点坐标 + 当前偏移量
            B = cubic_Bspline(img_coord / h - current_mesh_coord)  # 以第一维x为例, 括号中即为 x / h - i - 当前偏移量(共两维, 后面要把两个维度乘起来)
            impact = B.prod(axis=0, keepdim=True)  # 影响系数1 × H × W, dim=0相乘以实现B(x/h-i-偏移量)B(y/h-j-偏移量)
            i = current_mesh_coord[0].clamp(0, mesh.shape[-2] - 1)  # 当前节点距像素点左上角x-dim偏移量
            j = current_mesh_coord[1].clamp(0, mesh.shape[-1] - 1)  # 当前节点距像素点左上角y-dim偏移量
            key_mesh = mesh[:, i, j]  # 关键点位移cij
            shift += impact * key_mesh  # 像素点位移
        elif dim == 3:
            current_mesh_coord = mesh_coord + mesh_rel_coord.long().view(3, 1, 1, 1)
            B = cubic_Bspline(img_coord / h - current_mesh_coord)
            impact = B.prod(0, keepdim=True)
            i = current_mesh_coord[0].clamp(0, mesh.shape[-3] - 1)
            j = current_mesh_coord[1].clamp(0, mesh.shape[-2] - 1)
            k = current_mesh_coord[1].clamp(0, mesh.shape[-1] - 1)
            key_mesh = mesh[:, i, j, k]
            shift += impact * key_mesh
    return (img_coord + shift), shift  # 像素点对应的新坐标, 后续经过插值得到对应像素值
