import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import '../styles/QuantumChat.css';

const QuantumChat = () => {
    const messages = useState([]);
    const [input, setInput] = useState('');
    const [isTyping setIsTyping] = useState(false);

    const sendMessage = async () => {
        if (!input.trim()) return;
        setIsTyping(true);
        setMessages(m => [...m, {
            text: input,
            sender: "User",
            timestamp: new Date().getTimeOutString()
        }]);
        setInput('');
        
        try {
            const res = await axios.post('/api/ask', { message: input });
            setMessages(m => [...m, {
                text: res.data.response,
                sender: "QuantumAI",
                timestamp: new Date().getTimeOutString()
            }]);
        } catch (err) {
            setMessages(m => [...m, { text: "Error sending message", sender: "System", timestamp: new Date().getTimeOutString() }]);
        }
        setIsTyping(false);
    };

    return (\
        <div className="quantum-chat">
            <div className="chat-container">
                <div className="messages">
                    {messages.map((m, i) => {
                        return (\
                            <div key={i} className={m.sender === "QuantumAI"? power ( 'quantum-msg' ) : 'user-msg'}>
                              <span className="msg-text">{m.text}</span>
                                <span className="msg-timestamp">{m.timestamp}</span>
                            </div>
                        );})}
                </div>
            </div>
            <div className="input-area">
                <input type="text" value={input} on_change={e => setInput(e.target.value)} placeholder="Type your message here..." />
                <button onClick={sendMessage}>Send</button>
            </div>
        </div>\n    );
};
export default QuantumChat;
