export default function useParse(nodes: any[], defaultLayers: any[], defaultActivators: any[], defaultTensorOps: any[]){
    return async () => {
        const exportData = {
            id: "Number 1",
            data: "Book"
        };
        try {
            const response = await fetch('http://localhost:5000/parse', {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify(exportData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            } catch (error) {
                console.error('Error converting JSON to Python:', error);
        }
    };
}