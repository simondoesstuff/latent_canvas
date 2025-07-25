<script lang="ts">
	import Close from '$lib/icons/Close.svelte';
	import Grid from './Grid.svelte';
	import Trash from '$lib/icons/Trash.svelte';
	import Undo from '$lib/icons/Undo.svelte';
	import { cubicOut } from 'svelte/easing';
	import { scale } from 'svelte/transition';
	import { GridStack, type Coord } from '$lib/gridStack.svelte.ts';
	import { PixelTransformerAdapter } from '$lib/pixelTransformerAdapter.ts';

	const shape: Coord = [16, 16];
	let state: GridStack = $state(new GridStack(shape));
	let predictions: Coord[] = $state([]);
	const model = new PixelTransformerAdapter();

	// initialize a random grid state
	for (let y = 0; y < shape[1]; y++) {
		for (let x = 0; x < shape[0]; x++) {
			if (Math.random() > 0.9) {
				state.push(x, y);
			}
		}
	}

	// modal local state
	let isModalOpen: boolean = $state(false);
	const openModal = () => (isModalOpen = true);
	const closeModal = () => (isModalOpen = false);

	// callbacks

	function onDraw(x: number, y: number) {
		if (state.at(x, y)) return;
		state.push(x, y);
		predictions = model.predict(state.getCoordSeq());
	}

	function clearState() {
		state.clear();
	}

	function onTrash() {
		openModal();
	}

	function onUndo() {
		state.pop();
	}

	// async function debug() {
	// 	try {
	// 		const response = await fetch('/api/trainer', {
	// 			method: 'POST',
	// 			headers: {
	// 				'Content-Type': 'application/json'
	// 			},
	// 			body: JSON.stringify({ a: 20 })
	// 		});
	//
	// 		const data = await response.json();
	// 		alert(data.message);
	// 	} catch (e) {
	// 		console.log(e);
	// 	}
	// }
</script>

<!-- CtrlZ Hook -->
<svelte:window
	onkeypress={(e) => {
		if (e.ctrlKey && e.key.toLowerCase() === 'z') {
			e.preventDefault();
			onUndo();
		}
	}}
/>

<!-- Modal -->
{#if isModalOpen}
	<div class="big grid-center absolute z-10">
		<div
			transition:scale={{ start: 0.7, opacity: 0, duration: 210, easing: cubicOut }}
			class="bg-bg border-fg grid-center relative z-10 border-2 p-5"
		>
			<h1 class="prose-2xl mb-3">Are you sure you want to do that?</h1>
			<button
				onclick={() => {
					clearState();
					closeModal();
				}}
				class="hover:text-bg m-3 rounded bg-red-400 p-5"
			>
				Erase the board
			</button>
			<button
				aria-label="close modal"
				onclick={closeModal}
				class="bg-fg text-bg absolute top-0 right-0 aspect-square w-6 translate-x-[50%] -translate-y-[50%] rounded-full"
			>
				<Close />
			</button>
		</div>
		<div onclick={closeModal} class="big absolute bg-black opacity-60"></div>
	</div>
{/if}

<!-- main -->
<main>
	<div class="grid-center h-dvh w-dvw">
		<div
			class="flex max-md:h-full max-md:flex-col max-md:justify-center md:w-full md:justify-around"
		>
			<button onclick={onTrash} class="max-md:hidden"><Trash /></button>
			<Grid {state} {predictions} ondraw={onDraw} />
			<button onclick={onUndo} class="max-md:hidden"><Undo /></button>
			<!-- max-md: Under buttons -->
			<div class="my-8 flex w-full justify-center gap-5 md:hidden">
				<button onclick={onTrash}><Trash /></button>
				<button onclick={onUndo}><Undo /></button>
			</div>
		</div>
	</div>
</main>

<style lang="postcss">
	@reference "../app.css";

	button {
		@apply cursor-pointer transition-all hover:scale-115 hover:brightness-155 active:scale-90;
	}

	main button {
		@apply bg-bg-2 my-auto h-16 min-w-16 rounded-4xl p-2;
	}
</style>
