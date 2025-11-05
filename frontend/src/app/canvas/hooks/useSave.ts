export default function useSave(nodes: any[], edges: any[]) {
  return async () => {

    const exportData = {nodes, edges};
  // Accept both 'id' and 'network_id' for compatibility
  const params = new URLSearchParams(window.location.search);
  const id = params.get("id") || params.get("network_id");

    // DEBUG: Show the generated JSON structure
    console.log("=== GENERATED JSON STRUCTURE ===");
    console.log(JSON.stringify(exportData, null, 2));
    console.log("=== END DEBUG ===");
    
    // Send JSON to backend API to save the network
    // only thing different from useExport
    try {
      const response = await fetch('http://localhost:5000/save_network', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          nn_id: id || null, // must match backend param name
          name: `My Project #${id ? ' ' + id : ''}`,
          network: exportData
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Only redirect if it was a new network
        if (!id) {
          const savedId = result.id;
          window.location.href = `/canvas?id=${savedId}`;
          alert(`Network saved successfully!`);
        } else {
          // Optionally show a toast or message
          alert(`Network #${id} updated successfully!`);
        }
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error('Error saving network:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      alert(`Failed to save network: ${errorMessage}`);
    }
  };
}