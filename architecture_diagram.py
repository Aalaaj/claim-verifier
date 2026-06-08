from diagrams import Diagram, Edge, Cluster
from diagrams.programming.flowchart import Action, Document
from diagrams.onprem.workflow import Airflow
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.custom import Custom
from diagrams.aws.storage import SimpleStorageServiceS3Bucket

with Diagram(
    "RuleBasedClaimAnalyzer Architecture",
    filename="architecture",
    show=False,
    direction="LR",
    outformat=["png", "svg"],
    graph_attr={
        "fontsize": "18",
        "fontname": "Arial",
        "splines": "ortho",
        "nodesep": "0.5",
        "ranksep": "0.8",
    }
):
    
    # Input
    pdf = Document("PDF Input")
    
    # Stage 1: PDF Parsing (using Cluster for grouping)
    with Cluster("Stage 1: PDF Processing"):
        parser = Action("PDF Parser\n(PyMuPDF)")
        section_detection = Action("Section Detection\n(Abstract, Intro, Methods)")
        parser >> section_detection
    
    # Stage 2: Claim Processing
    with Cluster("Stage 2: Claim Processing", graph_attr={"bgcolor": "#E8F5E9"}):
        with Cluster("Detection Methods"):
            keyword = Action("Keyword Detector\n(Rule-based)")
            bert = Action("BERT Detector\n(ML - Future Work)")
            
        claim_classifier = Action("Claim Classification\n6 Types")
        
        # Connect
        keyword >> claim_classifier
        bert >> claim_classifier
    
    # Stage 3: Trust Calculation
    with Cluster("Stage 3: Trust Calculation", graph_attr={"bgcolor": "#FFF3E0"}):
        trust_calculator = Rack("Trust Calculator")
        
        # Individual formulas as separate nodes
        s_int = Action("S_int")
        s_ext = Action("S_ext")
        s_cit = Action("S_cit")
        s_meth = Action("S_meth")
        s_rep = Action("S_rep")
        
        # Group formulas
        trust_calculator >> s_int >> s_ext >> s_cit >> s_meth >> s_rep
    
    # Stage 4: Output
    output = SQL("Excel Export")
    
    # Weighted sum (as separate node)
    final_score = Action("T(C)")
    
    # Connect main pipeline
    pdf >> Edge(label="Upload", color="#666666", penwidth="2") >> parser
    section_detection >> Edge(label="Extract sentences", color="#666666", penwidth="2") >> keyword
    section_detection >> Edge(label="Extract sentences", color="#666666", penwidth="2", style="dashed") >> bert
    claim_classifier >> trust_calculator
    s_rep >> final_score
    final_score >> output
    