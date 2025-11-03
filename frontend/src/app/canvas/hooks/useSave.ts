export default function useSave(nodes: any[], edges: any[]) {
  return async () => {

    const exportData = {nodes, edges};

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
          name: 'Blah Blah Blah', // TODO: Get name from user input
          id: Math.floor(Math.random() * 1000000), // TODO: Generate or get a proper ID
          network: exportData
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        alert("Network saved successfully!");
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