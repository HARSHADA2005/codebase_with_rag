"""
AI handler module for OpenAI integration.
Manages chat completions and context-aware responses.
"""

from typing import List, Dict, Generator
import streamlit as st
from openai import OpenAI

import config
import utils


class AIHandler:
    """Handles interactions with OpenAI API."""
    
    def __init__(self, api_key: str):
        """
        Initialize the AI handler.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.model = config.DEFAULT_MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[str],
        chat_history: List[Dict] = None,
        stream: bool = None
    ) -> Generator[str, None, None] | str:
        """
        Generate a response using OpenAI with context from retrieved documents.
        
        Args:
            query: User query
            context_chunks: Relevant document chunks for context
            chat_history: Previous chat messages
            stream: Whether to stream the response
            
        Returns:
            Generator yielding response chunks if streaming, else complete response
        """
        stream = stream if stream is not None else config.STREAM_RESPONSE
        
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Build messages
        messages = self._build_messages(query, context, chat_history)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            utils.show_error(error_msg)
            if stream:
                return iter([error_msg])
            return error_msg
    
    def _build_context(self, chunks: List[str]) -> str:
        """
        Build context string from document chunks.
        
        Args:
            chunks: List of relevant text chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context available."
        
        context_parts = ["Here is the relevant context from the documents:\n"]
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Context {i}]\n{chunk}\n")
        
        return "\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        chat_history: List[Dict] = None
    ) -> List[Dict]:
        """
        Build message list for OpenAI API.
        
        Args:
            query: User query
            context: Context from documents
            chat_history: Previous messages
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that answers questions based on the provided context. "
                    "Always cite the context when answering. If the context doesn't contain enough information "
                    "to answer the question, say so clearly. Be concise but comprehensive."
                )
            }
        ]
        
        # Add chat history if available
        if chat_history:
            messages.extend(chat_history[-10:])  # Keep last 10 messages for context
        
        # Add current query with context
        user_message = f"{context}\n\nQuestion: {query}"
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """
        Stream response chunks from OpenAI.
        
        Args:
            response: OpenAI streaming response
            
        Yields:
            Response text chunks
        """
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def generate_summary(self, text: str) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise summaries."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide a concise summary of the following text:\n\n{text[:4000]}"
                    }
                ],
                max_tokens=200,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"


def create_chat_message(role: str, content: str) -> Dict:
    """
    Create a chat message dictionary.
    
    Args:
        role: Message role ('user' or 'assistant')
        content: Message content
        
    Returns:
        Message dictionary
    """
    return {
        "role": role,
        "content": content
    }
