import React from 'react';
import type { ActionType, KeyType } from '../types/GameTypes';
import { useGameState } from '../context/GameStateContext';
import GameService from '../service/GBService';

const Controls: React.FC = () => {
  const { setGameState } = useGameState();

  const action = (action: ActionType) => {
    GameService.sendAction(action).then(data => {
      setGameState(data);
    })
  }

  const handleKeyPress = (key: KeyType) => {
    console.log(`Key pressed: ${key}`);
    action({
      action_type: 'press_key',
      keys: [key]
    });
  };

  return (
    <div className="controls">
      <button onClick={() => handleKeyPress('up')}>Up</button>
      <button onClick={() => handleKeyPress('down')}>Down</button>
      <button onClick={() => handleKeyPress('left')}>Left</button>
      <button onClick={() => handleKeyPress('right')}>Right</button>
      <button onClick={() => handleKeyPress('a')}>A</button>
      <button onClick={() => handleKeyPress('b')}>B</button>
      <button onClick={() => handleKeyPress('start')}>Start</button>
      <button onClick={() => handleKeyPress('select')}>Select</button>
    </div>
  );
};

export default Controls;