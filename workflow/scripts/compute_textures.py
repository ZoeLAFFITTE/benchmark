import slam.io as sio
import slam.watershed as swat
import slam.texture as stex


white_mesh = sio.load_mesh(snakemake.input.mesh)

mean_curvature, dpf_star, _ = swat.compute_mesh_features(white_mesh)

### DPF_*
dpf_star = -dpf_star
dpf_star_tex = stex.TextureND(darray=dpf_star)
sio.write_texture(dpf_star_tex, snakemake.output.dpf)

### CURV
curv_tex = stex.TextureND(darray=mean_curvature)
sio.write_texture(curv_tex, snakemake.output.curv)
