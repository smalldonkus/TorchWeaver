import { useUser } from "@auth0/nextjs-auth0"
import { useCallback } from "react";

export default function useSave() {
  const { user, isLoading } = useUser();

  return useCallback(
    async(
      nodes: any[], 
      edges: any[],
      name: string,
      base64Image: string,
      onSuccess?: (message: string) => void,
      onError?: (message: string) => void,
    ) => {
      if (isLoading) {
        const message = "Loading user info...";
        if (onError) onError(message);
        else alert(message);
        return;
      }

      if (!user) {
        const message = "You must be logged in to save your network!";
        if (onError) onError(message);
        else alert(message);
        return;
      }

      const exportData = { nodes, edges };
      const params = new URLSearchParams(window.location.search);
      const id = params.get("id");

      try {
        const response = await fetch("http://localhost:5000/save_network", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            nn_id: id || null,
            name,
            network: exportData,
            preview: base64Image,
            user: {
              id: user.sub,
              name: user.name,
              email: user.email,
              picture: user.picture,
            },
          }),
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        if (result.success) {
          const successMessage = id
            ? `Network #${id} updated successfully!`
            : `Network saved successfully!`;
          if (onSuccess) onSuccess(successMessage);
          else alert(successMessage);

          if (!id) {
            setTimeout(() => (window.location.href = `/canvas?id=${result.id}`), 1500);
          }
        } else {
          const errorMsg = `Error: ${result.error}`;
          if (onError) onError(errorMsg);
          else alert(errorMsg);
        }
      } catch (error) {
        console.error("Error saving network:", error);
        const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
        const fullErrorMsg = `Failed to save network: ${errorMessage}`;
        if (onError) onError(fullErrorMsg);
        else alert(fullErrorMsg);
      }
    },
    [user, isLoading] // dependencies
  );
}