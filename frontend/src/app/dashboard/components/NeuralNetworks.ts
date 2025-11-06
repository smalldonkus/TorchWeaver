export type NeuralNetworkInfo = {
  id: number;
  title: string;
  lastAccessed: string;
  image: string;
  Owner: string;
  Favourited: boolean;
};

// Fetch list of networks from backend and transform to frontend type
export const getNeuralNetworks = async (userId?: string): Promise<NeuralNetworkInfo[]> => {
  try {
    const headers: Record<string, string> = {};
    if (userId) headers['header'] = userId;
    const res = await fetch('http://localhost:5000/list_network', { headers });
    if (!res.ok) {
      console.error('Failed to fetch networks', res.status);
      return [];
    }
    const data = await res.json();
    // backend returns { networks: [ {id, name, description, created_at} ] }
    return (data.networks || []).map((n: any) => ({
      id: n.id,
      title: `${n.name || `Network`} (ID: ${n.id})`,
      lastAccessed: n.created_at ? new Date(n.created_at).toLocaleString() : '',
      image: '/testnetwork.png',
      Owner: 'User',
      Favourited: false
    }));
  } catch (err) {
    console.error('Error fetching neural networks', err);
    return []; 
  }
};

export const loadNetwork = async (id: number, userId?: string) => {
  const headers: Record<string, string> = {};
  if (userId) headers['header'] = userId;
  const res = await fetch(`http://localhost:5000/load_network?id=${id}`, { headers });
  if (!res.ok) throw new Error('Failed to load network');
  const data = await res.json();
  return data.network;
};

export const deleteNetwork = async (id: number, userId?: string) => {
  const headers: Record<string, string> = {};
  if (userId) headers['header'] = userId;
  const res = await fetch(`http://localhost:5000/delete_network?id=${id}`, { method: 'DELETE', headers });
  if (!res.ok) throw new Error('Failed to delete network');
  return true;
};
