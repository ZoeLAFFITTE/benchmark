## 1. Mean curvature ----------------------------------------------------------
rule mindboggle_compute_mean_curvature:
    input:
        pial_vtk=f"{OUT}/{{test}}/{{sub}}/preproc/{{sub}}.{{side}}.{PIAL}.vtk"
    output:
        curv=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mean_curv.vtk"
    singularity:
        CONTAINERS["mindboggle"]
    
    log:f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mean_curv.log"

    benchmark:
        f"{OUT}/benchmarks/{{test}}/{{sub}}/{{sub}}.{{side}}.m1.benchmark.txt"

    shell:
        """
        /opt/vtk_cpp_tools/curvature/CurvatureMain \
            -m 2 \
            -n 0.75 \
            {input.pial_vtk} \
            {output.curv} \
            2>&1 | tee {log}
        """


## 2. Travel depth ------------------------------------------------------------
rule mindboggle_compute_travel_depth:
    input:
        pial_vtk=f"{OUT}/{{test}}/{{sub}}/preproc/{{sub}}.{{side}}.{PIAL}.vtk"

    output:
        depth=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.vtk"

    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.log"

    benchmark:
        f"{OUT}/benchmarks/{{test}}/{{sub}}/{{sub}}.{{side}}.m2.benchmark.txt"

    shell:
        """
        /opt/vtk_cpp_tools/travel_depth/TravelDepthMain \
            {input.pial_vtk} \
            {output.depth} \
            2>&1 | tee {log}
        """

## 3. Rescal depth ------------------------------------------------------------
rule mindboggle_compute_rescale_depth:
    input:
        depth=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.vtk"

    output:
        depth_rescaled=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth_rescaled.vtk"
    
    params:
        output_prefix=lambda wc:
            f"{OUT}/{wc.test}/{wc.sub}/mindboggle/{wc.sub}.{wc.side}.depth_rescaled"

    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth_rescaled.log"

    benchmark:
        f"{OUT}/benchmarks/{{test}}/{{sub}}/{{sub}}.{{side}}.m3.benchmark.txt"

    shell:
        """
        /opt/miniconda-latest/envs/mb/bin/python -u \
            ./scripts/mindboggle_compute_rescale_depth.py \
            {input.depth} \
            {params.output_prefix} \
            2>&1 | tee {log}
        """

## 4. Depth depth threshold ---------------------------------------------------
rule mindboggle_compute_depth_threshold:
    input:
        depth=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.vtk"
    
    output:
        folds_vtk=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.vtk",
        folds_npy=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.npy"
    
    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.log"

    benchmark:
        f"{OUT}/benchmarks/{{test}}/{{sub}}/{{sub}}.{{side}}.m4.benchmark.txt"
    
    shell:
        """
        /opt/miniconda-latest/envs/mb/bin/python -u \
            ./scripts/mindboggle_compute_folds.py \
            {input.depth} \
            {output.folds_vtk} \
            {output.folds_npy} \
            2>&1 | tee {log}
        """

## 5. Extract fundus ----------------------------------------------------------
rule mindboggle_compute_fundi:
    input:
        folds_npy=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.npy",
        curv=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mean_curv.vtk",
        depth_rescaled=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth_rescaled.vtk"
    
    output:
        fundi_vtk=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.fundi.vtk",
    
    singularity:
        CONTAINERS["mindboggle"]
    
    log: f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.fundi.log"

    benchmark:
        f"{OUT}/benchmarks/{{test}}/{{sub}}/{{sub}}.{{side}}.m5.benchmark.txt"
    
    shell:
        """
        /opt/miniconda-latest/envs/mb/bin/python -u \
            ./scripts/mindboggle_compute_fundi.py \
            {input.folds_npy} \
            {input.curv} \
            {input.depth_rescaled} \
            {output.fundi_vtk} \
            2>&1 | tee {log}
        """


rule mindboggle_aggregate_log:
    input:
        curv=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mean_curv.log",
        depth=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth.log",
        depth_rescaled=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.depth_rescaled.log",
        folds=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.folds.log",
        fundi=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.fundi.log"

    output:
        log=f"{OUT}/{{test}}/{{sub}}/mindboggle/{{sub}}.{{side}}.mindboggle.log"

    shell:
        """
        cat {input.curv} > {output.log}
        cat {input.depth} >> {output.log}
        cat {input.depth_rescaled} >> {output.log}
        cat {input.folds} >> {output.log}
        cat {input.fundi} >> {output.log}

        rm {input.curv} \
           {input.depth} \
           {input.depth_rescaled} \
           {input.folds} \
           {input.fundi} \
           2>&1 | tee {log}
        """