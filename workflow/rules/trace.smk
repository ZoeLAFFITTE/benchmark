rule trace:
    input:
        white_vtk=rules.fs_convert_gii_vtk.output.white_vtk,

    output:
        skeletons=f"{OUT}/{{test}}/{{sub}}/trace/{{sub}}.{{side}}.{WHITE}.scurve.bary",

    log: f"{OUT}/{{test}}/{{sub}}/trace/{{sub}}.{{side}}.{WHITE}.trace.log",

    params:
        output_prefix=lambda wc: f"{OUT}/{wc.test}/{wc.sub}/trace/{wc.sub}.{wc.side}.{WHITE}"

    singularity: CONTAINERS["trace"]

    message :
        "fundi extraction --> Trace() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})"

    benchmark:
        f"{OUT}/{{test}}/{{sub}}/trace/{{sub}}.{{side}}.{WHITE}.trace.benchmark.txt"

    threads: 5
    shell:
        """
        time CurveExtraction \
            --inputSurface {input.white_vtk} \
            --outputCurve {params.output_prefix} \
            --nThreads {threads} \
            --sulcus --bary --noVtk \
            2>&1 | tee {log}
        """

# 2. Trace -> GIFTII ------------------------------------------------------------
rule trace_convert_scalar_gii:
    input:
        white_gii = rules.fs_convert_gii_vtk.input.white_gii,
        skeletons=rules.trace.output.skeletons,

    output:
        skeletons=f"{OUT}/{{test}}/{{sub}}/trace/{{sub}}.{{side}}.skeletons.gii",

    conda: 
        ENVIRONMENTS["slam"]

    log:
        f"{OUT}/{{test}}/{{sub}}/trace/{{sub}}.{{side}}.trace_convert_scalar_gii.log",

    message:
        "postproc --> Trace/trace_convert_scalar_gii() (sub={wildcards.sub}, side={wildcards.side}, test={wildcards.test})",

    script:
        "../scripts/python/trace_convert_scalar_gii.py"