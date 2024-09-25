import axios from 'axios';

export default async function handler(req, res) {
  const { character, message, use_fine_tuned } = req.body;

  try {
    const flaskBackendUrl = process.env.FLASK_BACKEND_URL;

    if (!flaskBackendUrl) {
      throw new Error('FLASK_BACKEND_URL environment variable is not set');
    }

    const response = await axios.post(`${flaskBackendUrl}/api/generate`, {
      character,
      message,
      use_fine_tuned,
    },{
        timeout: 10000
    });

    res.status(200).json({ response: response.data.response });
  } catch (error) {
    console.error('Error generating generate AI response:', error);
    res.status(500).json({ error: 'Error generating generate AI response.' });
  }
}
