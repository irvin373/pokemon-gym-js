import { useGameState } from '../context/GameStateContext';
import Controls from './Controls';

const GameScreen = () => {
  const { gameState } = useGameState();

  if (!gameState) {
    return <div>Loading game state...</div>;
  }
  const { 
    player_name, 
    rival_name, 
    money, 
    location, 
    badges, 
    inventory, 
    dialog, 
    score 
  } = gameState;

  return (
    <div className="game-screen">
      <div className="game-info">
        <h2>Game Info</h2>
        <p><strong>Player:</strong> {player_name}</p>
        <p><strong>Rival:</strong> {rival_name}</p>
        <p><strong>Money:</strong> ${money}</p>
        <p><strong>Location:</strong> {location}</p>
        <p><strong>Badges:</strong> {badges.join(', ') || 'None'}</p>
        <p><strong>Inventory:</strong> {inventory.join(', ') || 'Empty'}</p>
        <p><strong>Dialog:</strong> {dialog || 'No dialog'}</p>
        <p><strong>Score:</strong> {score}</p>
      </div>
      <div className="game-map">
        <img src={`data:image/png;base64,${gameState.screenshot_base64}`} alt="Game Screenshot" />
      </div>
      <div className="pokemon-interactions">
        <Controls />
      </div>
    </div>
  );
};

export default GameScreen;