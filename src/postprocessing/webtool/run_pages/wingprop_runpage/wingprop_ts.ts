document.addEventListener('DOMContentLoaded', () => {
    const arrayForm = document.getElementById('arrayForm') as HTMLFormElement;
    const numArraysInput = document.getElementById('numArrays') as HTMLInputElement;
    const arrayInputsContainer = document.getElementById('arrayInputs') as HTMLDivElement;
    const addArrayButton = document.getElementById('addArray') as HTMLButtonElement;
    const runButton = document.getElementById('runButton') as HTMLButtonElement;
    const resultDiv = document.getElementById('result') as HTMLDivElement;
  
    addArrayButton.addEventListener('click', () => {
      const numArrays = parseInt(numArraysInput.value, 10);
      arrayInputsContainer.innerHTML = '';
  
      for (let i = 0; i < numArrays; i++) {
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = `Array ${i + 1} values (comma-separated)`;
        arrayInputsContainer.appendChild(input);
      }
    });
  
    runButton.addEventListener('click', async () => {
      const arrayInputs = Array.from(arrayInputsContainer.getElementsByTagName('input'));
      const arrayValues = arrayInputs.map(input => input.value.split(','));
  
      // Send arrayValues to the server and wait for the response
      const serverResponse = await sendRequestToServer(arrayValues);
  
      resultDiv.textContent = `Server Response: ${serverResponse}`;
    });
  
    async function sendRequestToServer(arrays: string[][]): Promise<string> {
      try {
        const response = await fetch('/run-script', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ scriptName: 'optimisation.py', arrays })
        });
  
        const data = await response.json();
        return data.result;
      } catch (error) {
        console.error('Error sending request to server:', error);
        return 'Error occurred';
      }
    }
  });
  