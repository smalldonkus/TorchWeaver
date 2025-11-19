import { useCallback, useRef } from 'react';
import { reconnectEdge, useReactFlow } from '@xyflow/react';

/**
 * Custom hook for handling edge reconnection functionality
 * Provides handlers for reconnecting and deleting edges
 */
export function useEdgeReconnection(setEdges: (edges: any[] | ((prevEdges: any[]) => any[])) => void) {
  const edgeReconnectSuccessful = useRef(true);
  
  const { deleteElements } = useReactFlow();

  const onReconnectStart = useCallback(() => {
    edgeReconnectSuccessful.current = false;
  }, []);

  const onReconnect = useCallback((oldEdge: any, newConnection: any) => {
    edgeReconnectSuccessful.current = true;
    setEdges((els: any[]) => reconnectEdge(oldEdge, newConnection, els));
  }, [setEdges]);

  const onReconnectEnd = useCallback((_: any, edge: any) => {
    if (!edgeReconnectSuccessful.current) {
      // setEdges((eds: any[]) => eds.filter((e) => e.id !== edge.id));
      deleteElements({edges: [{id: edge.id}]});
    }
    edgeReconnectSuccessful.current = true;
  }, [setEdges]);

  return {
    onReconnectStart,
    onReconnect,
    onReconnectEnd
  };
}