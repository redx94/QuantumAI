import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import { FaSmile, FaSpinner } from 'react-icons/fa';
import useSound from 'use-sound';

interface Message {
    id: string;
    text: string;
    isUser: boolean;
    timestamp: string;
    reactions?: string[];
}

export const QuantumChat: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const [playMessageSound] = useSound('/message-sound.mp3');

    const sendMessage = async () => {
        if (!input.trim()) return;
        
        const newMessage = {
            id: Date.now().toString(),
            text: input,
            isUser: true,
            timestamp: new Date().toLocaleTimeString(),
            reactions: []
        };
        
        playMessageSound();
        setMessages(prev => [...prev, newMessage]);
        setInput('');
        setLoading(true);
        setIsTyping(true);
        
        try {
            const response = await axios.post('/api/chat', { message: input });
            setIsTyping(false);
            playMessageSound();
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                text: response.data.response,
                isUser: false,
                timestamp: new Date().toLocaleTimeString(),
                reactions: []
            }]);
        } catch (error) {
            setIsTyping(false);
            setMessages(prev => [...prev, { text: 'Error processing message', isUser: false, timestamp: new Date().toLocaleTimeString() }]);
        }
        
        setLoading(false);
    };

    const addReaction = (messageId: string, reaction: string) => {
        setMessages(prev => prev.map(msg => 
            msg.id === messageId 
                ? { ...msg, reactions: [...(msg.reactions || []), reaction] }
                : msg
        ));
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="quantum-chat">
            <div className="messages-container">
                <AnimatePresence>
                    {messages.map((msg) => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className={`message ${msg.isUser ? 'user' : 'bot'}`}
                        >
                            <div className="message-content">
                                <ReactMarkdown>{msg.text}</ReactMarkdown>
                                <div className="message-meta">
                                    <span className="timestamp">{msg.timestamp}</span>
                                    <button 
                                        className="reaction-button"
                                        onClick={() => addReaction(msg.id, '👍')}
                                    >
                                        <FaSmile />
                                    </button>
                                </div>
                                {msg.reactions?.length > 0 && (
                                    <div className="reactions">
                                        {msg.reactions.map((reaction, i) => (
                                            <span key={i}>{reaction}</span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                    {isTyping && (
                        <motion.div 
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="typing-indicator"
                        >
                            <span>AI is thinking</span>
                            <FaSpinner className="spin" />
                        </motion.div>
                    )}
                </AnimatePresence>
                <div ref={messagesEndRef} />
            </div>
            <div className="input-area">
                <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && !loading && sendMessage()}
                    placeholder="Type your message... (Markdown supported)"
                    disabled={loading}
                />
                <button 
                    onClick={sendMessage} 
                    disabled={loading}
                    className={loading ? 'loading' : ''}
                >
                    {loading ? <FaSpinner className="spin" /> : 'Send'}
                </button>
            </div>
        </div>
    );
};
