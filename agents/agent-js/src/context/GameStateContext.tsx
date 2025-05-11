import React, { createContext, useContext, useState } from 'react';
import type { GameState } from '../types/GameTypes';

interface GameStateContextProps {
  gameState: GameState | null;
  setGameState: (state: GameState) => void;
}

const GameStateContext = createContext<GameStateContextProps | undefined>(undefined);

export const GameStateProvider = ({ children }) => {
  const [gameState, setGameState] = useState<GameState | null>(null);

  return (
    <GameStateContext.Provider value={{ gameState, setGameState }}>
      {children}
    </GameStateContext.Provider>
  );
};

export const useGameState = (): GameStateContextProps => {
  const context = useContext(GameStateContext);
  if (!context) {
    throw new Error('useGameState must be used within a GameStateProvider');
  }
  return context;
};
