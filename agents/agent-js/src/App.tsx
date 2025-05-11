import { useEffect } from 'react';
import Header from './components/Header';
import GameScreen from './components/GameScreen';
import { GameStateProvider, useGameState } from './context/GameStateContext';
import GameService from './service/GBService';
import './index.css';

function AppContent() {
  const { setGameState } = useGameState();

  useEffect(() => {
    GameService.startGame().then(data => {
      setGameState(data);
    });
  }, [setGameState]);

  return (
    <div className="app">
      <Header />
      <GameScreen />
    </div>
  );
}

function App() {
  return (
    <GameStateProvider>
      <AppContent />
    </GameStateProvider>
  );
}

export default App;