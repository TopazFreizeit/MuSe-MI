"""
Graph visualization utility.
"""
import logging
from typing import Any


def save_graph_visualization(graph: Any, output_path: str = "therapy_graph.png") -> None:
    """
    Generate and save a visual representation of the therapy graph.
    
    Args:
        graph: Compiled StateGraph
        output_path: Path where the image will be saved
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get the graph representation and draw it as PNG
        graph_image = graph.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(graph_image)
        
        logger.info(f"Graph visualization saved: {output_path}")
        
    except Exception as e:
        logger.warning(f"Visualization skipped: {str(e)}")
