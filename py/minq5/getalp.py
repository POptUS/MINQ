import numpy as np


def getalp(alpu, alpo, gTp, pTGp):
    """
    % function [alp,lba,uba,ier]=getalp(alpu,alpo,gTp,pTGp)
    % get minimizer alp in [alpu,alpo] for a univariate quadratic
    %	q(alp)=alp*gTp+0.5*alp^2*pTGp
    % lba	lower bound active
    % uba	upper bound active
    %
    % ier	 0 (finite minimizer)
    %	 1 (unbounded minimum)
    %
    """

    lba = 0
    uba = 0

    # determine unboundedness
    ier = 0
    if alpu == -np.inf and (pTGp < 0 or (pTGp == 0 and gTp > 0)):
        ier = 1
        lba = 1
    if alpo == np.inf and (pTGp < 0 or (pTGp == 0 and gTp < 0)):
        ier = 1
        uba = 1
    if ier:
        alp = NaN
        return alp, lba, uba, ier

    # determine activity
    if pTGp == 0 and gTp == 0:
        alp = 0
    elif pTGp <= 0:
        # concave case minimal at a bound
        if alpu == -np.inf:
            lba = 0
        elif alpo == np.inf:
            lba = 1
        else:
            lba = 2 * gTp + (alpu + alpo) * pTGp > 0
        uba = not lba
    else:
        alp = -gTp / pTGp  # unconstrained optimal step
        lba = alp <= alpu  # lower bound active
        uba = alp >= alpo  # upper bound active

    if lba:
        alp = alpu

    if uba:
        alp = alpo

    # print?
    if np.abs(alp) == np.inf:
        print(gTp, pTGp, alpu, alpo, alp, lba, uba, ier)

    return alp, lba, uba, ier
