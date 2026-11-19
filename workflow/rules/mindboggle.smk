## 1. Mean curvature ----------------------------------------------------------
rule mindboggle_compute_mean_curvature:
    input:
        pial_vtk=rules.fs_convert_gii_vtk.output.pial_vtk,
    output:
        curv_pial_vtk=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mean_curv.vtk",
    singularity:
        CONTAINERS["mindboggle"],
    
    log:f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle_compute_mean_curvature.log",

    benchmark:
        f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle1.benchmark.txt"
    
    message :
        "preprocessing --> Mindboggle/compute_mean_curvature() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})"

    shell:
        """
        /opt/vtk_cpp_tools/curvature/CurvatureMain \
            -m 2 \
            -n 0.75 \
            {input.pial_vtk} \
            {output.curv_pial_vtk} \
            2>&1 | tee {log}
        """


## 2. Travel depth ------------------------------------------------------------
rule mindboggle_compute_travel_depth:
    input:
        pial_vtk=rules.fs_convert_gii_vtk.output.pial_vtk

    output:
        depth_pial_vtk=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.vtk"

    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle_compute_travel_depth.log"

    benchmark:
        f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle2.benchmark.txt"
    
    message :
        "preprocessing --> Mindboggle/compute_travel_depth() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})"

    shell:
        """
        /opt/vtk_cpp_tools/travel_depth/TravelDepthMain \
            {input.pial_vtk} \
            {output.depth_pial_vtk} \
            2>&1 | tee {log}
        """

## 3. Rescal depth ------------------------------------------------------------
rule mindboggle_compute_rescale_depth:
    input:
        depth_pial_vtk=rules.mindboggle_compute_travel_depth.output.depth_pial_vtk

    output:
        depth_pial_vtk_res=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth_rescaled.vtk"
    
    params:
        output_prefix=lambda wc:
            f"{OUT}/{wc.test}/{wc.sub}/mindboggle/{wc.sub}.{wc.side}.depth_rescaled"

    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle_compute_rescale_depth.log"

    benchmark:
        f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle3.benchmark.txt"
    
    message :
        "preprocessing --> Mindboggle/compute_rescale_depth() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})"

    shell:
        """
        /opt/miniconda-latest/envs/mb/bin/python -u \
            ./scripts/mindboggle_compute_rescale_depth.py \
            {input.depth_pial_vtk} \
            {params.output_prefix} \
            2>&1 | tee {log}
        """

## 4. Depth depth threshold ---------------------------------------------------
rule mindboggle_compute_fundi:
    input:
        curv_pial_vtk=rules.mindboggle_compute_mean_curvature.output.curv_pial_vtk,
        depth_pial_vtk=rules.mindboggle_compute_travel_depth.output.depth_pial_vtk,
        depth_pial_vtk_res=rules.mindboggle_compute_rescale_depth.output.depth_pial_vtk_res,
    
    output:
        folds_pial_vtk=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.vtk",
        skeletons_vtk=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.skeletons.vtk",
    
    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle_compute_fundi.log"

    benchmark:
        f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{PIAL}.mindboggle4.benchmark.txt"
    
    message :
        "fundi extraction --> Mindboggle() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})"
    
    shell:
        """
        /opt/miniconda-latest/envs/mb/bin/python -u \
            ./scripts/mindboggle_compute_folds.py \
            {input.curv_pial_vtk} \
            {input.depth_pial_vtk} \
            {input.depth_pial_vtk_res} \
            {output.folds_pial_vtk} \
            {output.skeletons_vtk} \
            2>&1 | tee {log}
        """


# 5. VTIK -> GIFTII ------------------------------------------------------------
rule mindboggle_convert_gii:
    input:
        white_gii=rules.fs_convert_gii_vtk.input.white_gii,
        curv_pial_vtk=rules.mindboggle_compute_mean_curvature.output.curv_pial_vtk,
        depth_pial_vtk=rules.mindboggle_compute_travel_depth.output.depth_pial_vtk,
        depth_pial_vtk_res=rules.mindboggle_compute_rescale_depth.output.depth_pial_vtk_res,
        folds_pial_vtk=rules.mindboggle_compute_fundi.output.folds_pial_vtk,
        skeletons_vtk=rules.mindboggle_compute_fundi.output.skeletons_vtk,

    output:
        curv_pial_gii=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.{WHITE}.mean_curv.gii",
        depth_pial_gii=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.gii",
        depth_pial_gii_res=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth_rescaled.gii",
        folds_pial_gii=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.gii",
        skeletons=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.skeletons.gii",

    conda: ENVIRONMENTS["slam"]

    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mindboggle_convert_gii.log"

    message :
        "postproc--> Mindboggle/convert_gii() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})"

    script:
        f"../scripts/python/mindboggle_convert_scalar_gii.py"


rule mindboggle_aggregate_log:
    input:
        curv=rules.mindboggle_compute_mean_curvature.log,
        depth=rules.mindboggle_compute_travel_depth.log,
        depth_rescaled=rules.mindboggle_compute_rescale_depth.log,
        skeletons=rules.mindboggle_compute_fundi.log,
        convert=rules.mindboggle_convert_gii.log,

    output:
        log=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mindboggle.log"

    shell:
        """
        cat {input.curv} > {output.log}
        cat {input.depth} >> {output.log}
        cat {input.depth_rescaled} >> {output.log}
        cat {input.skeletons} >> {output.log}
        cat {input.convert} >> {output.log}

        rm {input.curv} \
           {input.depth} \
           {input.depth_rescaled} \
           {input.skeletons} \
           {input.convert} \
           2>&1 | tee {log}
        """