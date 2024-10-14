import axios from 'axios';

export default async function handler(req, res) {
  if (req.method === 'POST') {
    const { character, message } = req.body; // Removed use_fine_tuned

    try {
      // Send the data to the backend (FastAPI or Flask)
      const response = await axios.post('http://127.0.0.1:8000/api/generate', {
        character,
        message,
      });

      // Pass the response back to the frontend
      res.status(200).json({ response: response.data.response });
    } catch (error) {
      console.error('Error getting response from backend:', error);
      res.status(500).json({ error: 'Error generating response.' });
    }
  } else {
    // If it's not a POST request, return a method not allowed error
    res.setHeader('Allow', ['POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}
