function drift = evaluatePolynomialDrift(state, B, C, D)
drift = squeeze(sum(sum(B .* (state * state.'), 1), 2)) + C * state + D;
end
