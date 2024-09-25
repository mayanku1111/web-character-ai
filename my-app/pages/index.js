import { useState, useRef } from 'react';
import axios from 'axios';

export default function Home() {
  const [character, setCharacter] = useState({
    name: '',
    tagline: '',
    description: '',
    greeting: '',
    isPublic: true,
    useFinetuning: false,
  });
  const [showChat, setShowChat] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const inputRef = useRef(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCharacter((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('Submitting character:', character);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setShowChat(true);
  };

  const sendMessage = async (message) => {
    setChatMessages((prev) => [...prev, { role: 'user', content: message }]);
    const aiResponse = await getAIResponse(message, character);
    setChatMessages((prev) => [...prev, { role: 'assistant', content: aiResponse }]);
  };

  const getAIResponse = async (message, character) => {
    try {
      const response = await axios.post('/api/generate', {
        character,
        message,
        use_fine_tuned: character.useFinetuning,
      });
      return response.data.response;
    } catch (error) {
      console.error('Error getting final AI response:', error);
      return 'Error getting final AI response.';
    }
  };

  const styles = {
    container: {
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'Poppins, sans-serif',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: '#ECECF1',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
      borderRadius: '8px',
    },
    header: {
      fontSize: '28px',
      fontWeight: 'bold',
      marginBottom: '20px',
      textAlign: 'center',
    },
    form: {
      display: 'flex',
      flexDirection: 'column',
      gap: '15px',
    },
    label: {
      fontWeight: 'bold',
      marginBottom: '5px',
    },
    input: {
      width: '100%',
      padding: '10px',
      borderRadius: '8px',
      border: '1px solid #565869',
      backgroundColor: '#40414F',
      color: '#ECECF1',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    },
    textarea: {
      width: '100%',
      padding: '10px',
      borderRadius: '8px',
      border: '1px solid #565869',
      backgroundColor: '#40414F',
      color: '#ECECF1',
      minHeight: '100px',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    },
    switchContainer: {
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
    },
    switchLabel: {
      fontWeight: 'bold',
    },
    toggleSwitch: {
      position: 'relative',
      display: 'inline-block',
      width: '40px',
      height: '20px',
    },
    toggleSlider: {
      position: 'absolute',
      cursor: 'pointer',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: '#ccc',
      transition: '.4s',
      borderRadius: '20px',
    },
    toggleSliderChecked: {
      backgroundColor: '#10A37F',
    },
    toggleThumb: {
      position: 'absolute',
      height: '18px',
      width: '18px',
      left: '1px',
      bottom: '1px',
      backgroundColor: '#fff',
      transition: '.4s',
      borderRadius: '50%',
    },
    toggleThumbChecked: {
      transform: 'translateX(20px)',
    },
    button: {
      padding: '10px 20px',
      backgroundColor: '#10A37F',
      color: '#ffffff',
      border: 'none',
      borderRadius: '8px',
      cursor: 'pointer',
      fontWeight: 'bold',
      transition: 'background-color 0.3s',
    },
    chatContainer: {
      flex: 1,
      overflowY: 'auto',
      marginBottom: '20px',
    },
    chatMessage: {
      padding: '15px',
      marginBottom: '10px',
      borderRadius: '8px',
    },
    userMessage: {
      backgroundColor: '#343541',
      color: '#ECECF1',
    },
    assistantMessage: {
      backgroundColor: '#444654',
      color: '#ECECF1',
    },
    inputContainer: {
      position: 'relative',
      marginTop: '20px',
    },
    chatInput: {
      width: '100%',
      padding: '15px',
      paddingRight: '40px',
      borderRadius: '8px',
      border: '1px solid #565869',
      backgroundColor: '#40414F',
      color: '#ECECF1',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    },
    sendButton: {
      position: 'absolute',
      right: '10px',
      top: '50%',
      transform: 'translateY(-50%)',
      background: 'none',
      border: 'none',
      color: '#ECECF1',
      cursor: 'pointer',
    },
  };

  const ToggleSwitch = ({ checked, onChange }) => (
    <div style={styles.toggleSwitch} onClick={onChange}>
      <div
        style={{
          ...styles.toggleSlider,
          ...(checked ? styles.toggleSliderChecked : {}),
        }}
      >
        <div
          style={{
            ...styles.toggleThumb,
            ...(checked ? styles.toggleThumbChecked : {}),
          }}
        />
      </div>
    </div>
  );

  if (showChat) {
    return (
      <div style={styles.container}>
        <h2 style={styles.header}>Chat with {character.name}</h2>
        <div style={styles.chatContainer}>
          {chatMessages.map((msg, index) => (
            <div
              key={index}
              style={{
                ...styles.chatMessage,
                ...(msg.role === 'user' ? styles.userMessage : styles.assistantMessage),
              }}
            >
              {msg.content}
            </div>
          ))}
        </div>
        <div style={styles.inputContainer}>
          <input
            ref={inputRef}
            style={styles.chatInput}
            placeholder="Type your message..."
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                sendMessage(e.target.value);
                e.target.value = '';
              }
            }}
          />
          <button
            style={styles.sendButton}
            onClick={() => {
              if (inputRef.current.value) {
                sendMessage(inputRef.current.value);
                inputRef.current.value = '';
              }
            }}
          >
            ▶
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>Create a Character</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <div>
          <label style={styles.label}>Character Name</label>
          <input
            style={styles.input}
            name="name"
            value={character.name}
            onChange={handleInputChange}
            placeholder="e.g. Albert Einstein"
          />
        </div>
        <div>
          <label style={styles.label}>Tagline</label>
          <input
            style={styles.input}
            name="tagline"
            value={character.tagline}
            onChange={handleInputChange}
            placeholder="Add a short tagline of your Character"
          />
        </div>
        <div>
          <label style={styles.label}>Description</label>
          <textarea
            style={styles.textarea}
            name="description"
            value={character.description}
            onChange={handleInputChange}
            placeholder="How would your Character describe themselves?"
          />
        </div>
        <div>
          <label style={styles.label}>Greeting</label>
          <input
            style={styles.input}
            name="greeting"
            value={character.greeting}
            onChange={handleInputChange}
            placeholder="e.g. Hello, I am Albert. Ask me anything about my scientific contributions."
          />
        </div>
        <div style={styles.switchContainer}>
          <label style={styles.switchLabel}>Make this character public</label>
          <ToggleSwitch
            checked={character.isPublic}
            onChange={() =>
              setCharacter((prev) => ({ ...prev, isPublic: !prev.isPublic }))
            }
          />
        </div>
        <div style={styles.switchContainer}>
          <label style={styles.switchLabel}>Use fine-tuning</label>
          <ToggleSwitch
            checked={character.useFinetuning}
            onChange={() =>
              setCharacter((prev) => ({ ...prev, useFinetuning: !prev.useFinetuning }))
            }
          />
        </div>
        <button style={styles.button} type="submit">
          Create Character
        </button>
      </form>
    </div>
  );
}
