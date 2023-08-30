const express = require('express');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 3000;

app.use(express.json());

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'wingprop_runpage')));

app.post('/run-script', (req, res) => {
  const { arrays } = req.body;

  const pythonProcess = spawn('python', [path.join(__dirname, '..', '..', 'examples', 'optimisation', 'wingonly_optimisation.py'), JSON.stringify(arrays)]);

  let result = '';

  pythonProcess.stdout.on('data', data => {
    result += data.toString();
  });

  pythonProcess.on('close', code => {
    if (code === 0) {
      res.json({ result });
    } else {
      res.status(500).json({ error: 'Script execution failed' });
    }
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
