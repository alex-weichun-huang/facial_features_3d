import torch


def decode(emoca, values, training=False):
    with torch.no_grad():
        values = emoca.decode(values, training=training)
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        # load template and pass it to the visualization function
        visualizations, grid_image = emoca._visualization_checkpoint(
            values['verts'],
            values['trans_verts'],
            values['ops'],
            uv_detail_normals,
            values, 
            0,
            "",
            "",
            save=False
        )

    return values, visualizations


def test(deca, img, device = "cuda"):
    img["image"] = img["image"].to(device)
    deca = deca.to(device)
    if len(img["image"].shape) == 3:
        img["image"] = img["image"].view(1,3,224,224)
    vals = deca.encode(img, training=False)
    vals, visdict = decode(deca, vals, training=False)
    return vals, visdict

def batch_test(deca, batch_imgs, device = "cuda"):
    img_dict = {"image":batch_imgs} # required by the deca encode model
    deca = deca.to(device)
    vals = deca.encode(img_dict, training=False)
    vals, visdict = decode(deca, vals, training=False)
    return vals, visdict