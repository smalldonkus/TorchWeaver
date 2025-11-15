import { useState, useEffect } from "react";

export interface NodeDefinition {
  type: string;
  parameters_format: Record<string, string[]>;
  class?: string;
}

export interface HierarchicalNodeData {
  data: Record<string, Record<string, any>>;
}

export type NodeType = 'layers' | 'tensorops' | 'activators' | 'inputs';

export const useNodeDefinitions = (nodeType: NodeType) => {
  const [nodeData, setNodeData] = useState<HierarchicalNodeData | null>(null);
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
        
        setNodeData(data);
        setLoading(false);
      } catch (err) {
        console.error(`Error fetching ${nodeType} definitions:`, err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    };

    fetchNodeDefinitions();
  }, [nodeType]);

  const findNodeDefinition = (nodeTypeName: string): NodeDefinition | null => {
    if (!nodeData?.data) return null;
    
    // Search through all classes to find the node type
    for (const [className, classItems] of Object.entries(nodeData.data)) {
      if (classItems[nodeTypeName]) {
        return {
          type: nodeTypeName,
          parameters_format: classItems[nodeTypeName].parameters_format || {},
          class: className,
          ...classItems[nodeTypeName]
        };
      }
    }
    return null;
  };

  const getParameterFormat = (nodeTypeName: string, parameterName: string): string[] => {
    const node = findNodeDefinition(nodeTypeName);
    return node?.parameters_format[parameterName] || [];
  };

  return {
    nodeData,
    findNodeDefinition,
    getParameterFormat,
    loading,
    error
  };
};