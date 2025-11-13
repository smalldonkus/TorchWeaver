import { useUser } from "@auth0/nextjs-auth0"

export default function useSave(
  nodes: any[], 
  edges: any[],
  onSuccess?: (message: string) => void,
  onError?: (message: string) => void
) {
  const { user, isLoading } = useUser();

  return async () => {

    if (isLoading) {
      const message = "Loading user info...";
      if (onError) {
        onError(message);
      } else {
        alert(message);
      }
      return;
    }

    if (!user) {
      const message = "You must be logged in to save your network!";
      if (onError) {
        onError(message);
      } else {
        alert(message);
      }
      return;
    }

    const exportData = {nodes, edges};
    // Accept both 'id' and 'network_id' for compatibility
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");

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
          network: exportData,
          user: {
            id: user.sub,
            name: user.name,
            email: user.email,
            picture: user.picture,
          },
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
          const successMessage = `Network saved successfully!`;
          if (onSuccess) {
            onSuccess(successMessage);
          } else {
            alert(successMessage);
          }
          // Delay redirect slightly to show the snackbar
          setTimeout(() => {
            window.location.href = `/canvas?id=${savedId}`;
          }, 1500);
        } else {
          // Optionally show a toast or message
          const successMessage = `Network #${id} updated successfully!`;
          if (onSuccess) {
            onSuccess(successMessage);
          } else {
            alert(successMessage);
          }
        }
      } else {
        const errorMsg = `Error: ${result.error}`;
        if (onError) {
          onError(errorMsg);
        } else {
          alert(errorMsg);
        }
      }
    } catch (error) {
      console.error('Error saving network:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      const fullErrorMsg = `Failed to save network: ${errorMessage}`;
      if (onError) {
        onError(fullErrorMsg);
      } else {
        alert(fullErrorMsg);
      }
    }
  };
}