// Demo for toon.ts (TypeScript)
import { toToon, fromToon, buildPromptForLLM, chunkToon } from './toon';

const sample = [
  { id: 1, name: 'Widget A', price: 9.99, available: true },
  { id: 2, name: 'Widget B', price: 15.5, available: false },
];

console.log('--- TOON ---');
console.log(toToon(sample, undefined, '|', undefined, true));
console.log('--- Parsed back ---');
console.log(fromToon(toToon(sample, undefined, '|', undefined, true)));

console.log('--- Prompt ---');
console.log(buildPromptForLLM('Return ids and names for items with price > 10', sample));

console.log('--- Chunks ---');
console.log(chunkToon(Array.from({ length: 10 }).map((_, i) => ({ id: i, v: i * 2 })), 3));
