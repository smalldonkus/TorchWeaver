export default function useExport(nodes: any[], edges: any[]) {
  return () => {
    const inputMap: Record<string, string[]> = {};
    edges.forEach((edge) => {
      if (!inputMap[edge.target]) inputMap[edge.target] = [];
      inputMap[edge.target].push(edge.source);
    });

    const layers = nodes.map((node) => {
      const [typeRaw, ...labelParts] = (node.data.label || "").split(":");
      const type = typeRaw.trim().toLowerCase();
      const id = labelParts.join(":").trim() || node.id;
      const params = node.data.parameters || {};
      let parameters: any = {};

      if (type === "linear") {
        parameters = {
          in_features: Number(params.in_features) || null,
          out_features: Number(params.out_features) || null,
        };
      }

      return {
        id,
        type,
        inputs: (inputMap[node.id] || []).map((src) => `${src}_out`),
        outputs: [`${node.id}_out`],
        parameters,
      };
    });

    const json = JSON.stringify({ layers }, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "diagram.json";
    link.click();
    URL.revokeObjectURL(url);
  };
}
