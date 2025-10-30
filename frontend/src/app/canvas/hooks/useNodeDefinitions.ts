import { useState, useEffect } from "react";

export interface NodeDefinition {
  type: string;
  parameters_format: Record<string, string[]>;
}

export type NodeType = 'layers' | 'tensorops' | 'activators';

export const useNodeDefinitions = (nodeType: NodeType) => {
  const [nodeDefinitions, setNodeDefinitions] = useState<NodeDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNodeDefinitions = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/operations/${nodeType}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch ${nodeType} definitions`);
        }
        const data = await response.json();
        setNodeDefinitions(data.data || []);
        setLoading(false);
      } catch (err) {
        console.error(`Error fetching ${nodeType} definitions:`, err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    };

    fetchNodeDefinitions();
  }, [nodeType]);

  const getParameterFormat = (nodeTypeName: string, parameterName: string): string[] => {
    const node = nodeDefinitions.find(n => n.type === nodeTypeName);
    return node?.parameters_format[parameterName] || [];
  };

  return {
    nodeDefinitions,
    getParameterFormat,
    loading,
    error
  };
};