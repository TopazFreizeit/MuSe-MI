"""
Generate graph visualizations for single and multi-session therapy graphs.
"""
import logging
import sys
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graphs import SingleSessionGraphBuilder, MultiSessionGraphBuilder
from src.graphs.visualization import save_graph_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Generating graph visualizations...")
    
    # Build single-session graph
    single_builder = SingleSessionGraphBuilder()
    single_graph = single_builder.build()
    logger.info("Single-session graph built")
    
    # Save single-session graph visualization
    save_graph_visualization(single_graph, "single_session_graph.png")
    logger.info("Single-session graph visualization saved: single_session_graph.png")
    
    # Build multi-session graph with single-session graph as dependency
    multi_builder = MultiSessionGraphBuilder(single_graph)
    multi_graph = multi_builder.build()
    logger.info("Multi-session graph built")
    
    # Save multi-session graph visualization
    save_graph_visualization(multi_graph, "multi_session_graph.png")
    logger.info("Multi-session graph visualization saved: multi_session_graph.png")
    
    logger.info("Visualization generation complete!")
    print("\n" + "=" * 80)
    print("VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print("Files created:")
    print("  - single_session_graph.png")
    print("  - multi_session_graph.png")
    print("=" * 80)
