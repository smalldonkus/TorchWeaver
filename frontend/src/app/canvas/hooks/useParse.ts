export default async function useParse(nodes: any[], edges: any[]): Promise<any> {
    
    // Note: The parsing is now handled on the backend with the new hierarchical structure
    // The backend parse.py has been updated to handle:
    // - Layers.json (with global classes like "Linear Layers", "Dropout Layers")  
    // - TensorOperations.json (with global classes like "Merge Tensor Operations", "Split Tensor Operations")
    // - ActivationFunction.json (with global classes like "Activation Functions")
    
    // filters nodesList for only stuff needed,
    // fetches incoming/outgoing edges by ID only
    const exportNodes = nodes.map((n) => ({
        id: n.id, 
        data: n.data,
        parents:  edges.filter((e) => e.target == n.id).map((o) => o.source),
        children: edges.filter((e) => e.source == n.id).map((o) => o.target)
    }));

    const bodyData = {
        nodes : exportNodes
    }

    try {
            const response = await fetch('http://localhost:5000/api/parse', {
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
        );
         
        } catch (error) {
            console.error('Error converting JSON to Python:', error);
    }
}

