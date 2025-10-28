import { error } from "console";

 export default async function useParse(nodes: any[], edges: any[]): Promise<any> {
    
    // const defaultNodes = [
    //     ...defaultLayers,
    //     ...defaultActivators,
    //     ...defaultTensorOps
    // ]
    // filters nodesList for only stuff needed,
    // fetches incoming/outgoing edges by ID only
    const exportNodes = nodes.map((n) => ({
        id: n.id, data: n.data,
        parents:  edges.filter((e) => e.target == n.id).map((e) => e.id),
        children: edges.filter((e) => e.source == n.id).map((e) => e.id)
    }));

    const bodyData = {
        nodes : exportNodes
    }

    try {
        const response = await fetch('http://127.0.0.1:5000/parser/parse', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify(bodyData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        return new Promise( (resolve) => {
                setTimeout(() => {
                        resolve(result.info)
                    }, 100
                );
            }
        )
         
        } catch (error) {
            console.error('Error converting JSON to Python:', error);
    }
}

